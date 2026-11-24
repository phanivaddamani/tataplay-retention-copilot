"""copilot.py -- The Retention Copilot brief generator."""
import os
import sys
import json
import sqlite3
import joblib
import pandas as pd
import numpy as np

from typing import Literal
from pydantic import BaseModel, ValidationError
from langchain_core.output_parsers import PydanticOutputParser

import chromadb
from chromadb.utils import embedding_functions

import mock_llm
from llm_client import call_llm
from src.features import build_features, NUMERIC_FEATURES, CATEGORICAL_FEATURES

DB_PATH = os.path.join("data", "tataplay.db")
POL_DIR = os.path.join("data", "policies")
OUTPUT_DIR = "outputs"
CHURN_MODEL_PATH = os.path.join(OUTPUT_DIR, "churn_model.joblib")
SEGMENTS_PATH = os.path.join(OUTPUT_DIR, "segments.csv")
PRIORITY_PATH = os.path.join(OUTPUT_DIR, "retention_priority.csv")

class RetentionBrief(BaseModel):
    subscriber_id: str
    risk_band: Literal["High", "Medium", "Low"]
    recommended_action: str
    policy_source: str

def chunk_text(text, size=400, overlap=60):
    chunks = []
    step = size - overlap
    start = 0
    while start + overlap < len(text):
        chunks.append(text[start:start+size])
        start += step
    return chunks

# Module-level cache for policy vector collection and loaded model
_collection = None
_model = None

def get_policy_collection():
    global _collection
    if _collection is not None:
        return _collection
        
    client = chromadb.Client()
    ef = embedding_functions.ONNXMiniLM_L6_V2()
    coll_name = "copilot_policies"
    try:
        client.delete_collection(coll_name)
    except Exception:
        pass
    collection = client.create_collection(
        name=coll_name,
        metadata={"hnsw:space": "cosine"},
        embedding_function=ef
    )
    
    files = sorted(os.listdir(POL_DIR))
    all_chunks = []
    for fname in files:
        with open(os.path.join(POL_DIR, fname), "r", encoding="utf-8") as f:
            txt = f.read().strip()
        f_chunks = chunk_text(txt, size=400, overlap=60)
        for i, ch in enumerate(f_chunks):
            all_chunks.append({
                "id": f"{fname}_{i}",
                "source": fname,
                "chunk": i,
                "text": ch
            })
            
    collection.add(
        ids=[c["id"] for c in all_chunks],
        documents=[c["text"] for c in all_chunks],
        metadatas=[{"source": c["source"], "chunk": c["chunk"]} for c in all_chunks]
    )
    _collection = collection
    return _collection

def get_churn_model():
    global _model
    if _model is not None:
        return _model
    _model = joblib.load(CHURN_MODEL_PATH)
    return _model

def build_brief(subscriber_id):
    """
    Builds a verified retention brief for a given subscriber_id.
    
    Steps:
      (i) Read subscriber and recharge history from SQLite, compute features, replace None with np.nan.
      (ii) Score with saved churn model, verify against recorded score, derive risk band.
      (iii) Read segment from outputs/segments.csv.
      (iv) Retrieve top 3 policy chunks.
      (v) Call mock LLM with grounded RETENTION BRIEF prompt.
      (vi) Validate output with Pydantic RetentionBrief and verify consistency.
    """
    conn = sqlite3.connect(DB_PATH)
    sub_df = pd.read_sql_query("SELECT * FROM subscribers WHERE subscriber_id = ?", conn, params=[subscriber_id])
    rech_df = pd.read_sql_query("SELECT * FROM recharges WHERE subscriber_id = ?", conn, params=[subscriber_id])
    conn.close()
    
    if len(sub_df) == 0:
        raise ValueError(f"Subscriber {subscriber_id} not found in database.")
        
    # Replace None with NaN so categorical imputer / pipeline behaves consistently
    sub_df = sub_df.replace({None: np.nan})
    rech_df = rech_df.replace({None: np.nan})
    
    # Build full feature set as of 2026-03-31
    feat_df = build_features(sub_df, rech_df, snapshot_date="2026-03-31")
    feat_df = feat_df.replace({None: np.nan})
    
    # Cast integers
    tenure_months = int(feat_df['tenure_months'].iloc[0])
    failed_recharges_6m = int(feat_df['n_failed_6m'].iloc[0])
    
    # (ii) Model prediction & risk band
    model = get_churn_model()
    X_sub = feat_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    rescore_prob = float(model.predict_proba(X_sub)[:, 1][0])
    
    # Read churn_probability from priority or test_scores file if available
    prio_df = pd.read_csv(PRIORITY_PATH)
    match_row = prio_df[prio_df['subscriber_id'] == subscriber_id]
    if len(match_row) > 0:
        file_prob = float(match_row['churn_probability'].iloc[0])
        # Assert agreement to 4 decimals
        assert round(rescore_prob, 4) == round(file_prob, 4), f"Prob mismatch: {rescore_prob:.4f} vs {file_prob:.4f}"
        churn_prob = file_prob
    else:
        churn_prob = rescore_prob
        
    if churn_prob >= 0.40:
        risk_band = "High"
    elif churn_prob >= 0.20:
        risk_band = "Medium"
    else:
        risk_band = "Low"
        
    # (iii) Segment name
    seg_df = pd.read_csv(SEGMENTS_PATH)
    seg_match = seg_df[seg_df['subscriber_id'] == subscriber_id]
    segment_name = seg_match['segment_name'].iloc[0] if len(seg_match) > 0 else "Unknown"
    
    # (iv) Retrieve top 3 policy chunks
    query = f"retention offer for a {risk_band} risk subscriber with {tenure_months} months tenure and {failed_recharges_6m} failed recharges"
    coll = get_policy_collection()
    res = coll.query(query_texts=[query], n_results=3)
    
    context_blocks = []
    for doc, meta in zip(res["documents"][0], res["metadatas"][0]):
        context_blocks.append(f"[{meta['source']}]\n{doc}")
    context_str = "\n\n".join(context_blocks)
    
    # (v) Prompt LLM
    sys_msg = (
        "RETENTION BRIEF. You are a retention desk specialist. "
        "Generate a structured retention brief in valid JSON format only, using only the provided policy context."
    )
    user_msg = (
        f"FACTS:\n"
        f"subscriber_id: {subscriber_id}\n"
        f"risk_band: {risk_band}\n"
        f"tenure_months: {tenure_months}\n"
        f"failed_recharges_6m: {failed_recharges_6m}\n"
        f"segment: {segment_name}\n\n"
        f"POLICY CONTEXT:\n"
        f"{context_str}"
    )
    
    parser = PydanticOutputParser(pydantic_object=RetentionBrief)
    messages = [{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}]
    
    attempts = 0
    validated_brief = None
    while attempts < 3:
        attempts += 1
        reply = call_llm(messages)
        try:
            # Parse and validate JSON
            brief_obj = parser.parse(reply)
            
            # Consistency checks
            assert brief_obj.subscriber_id == subscriber_id, f"ID mismatch: {brief_obj.subscriber_id} vs {subscriber_id}"
            assert brief_obj.risk_band == risk_band, f"Risk band mismatch: {brief_obj.risk_band} vs {risk_band}"
            valid_sources = ["retention_offers_policy.txt", "none"]
            assert brief_obj.policy_source in valid_sources, f"Invalid policy source: {brief_obj.policy_source}"
            
            validated_brief = brief_obj
            break
        except Exception as e:
            messages.append({"role": "assistant", "content": reply})
            messages.append({"role": "user", "content": f"VALIDATION ERROR: {e}"})
            
    if validated_brief is None:
        raise RuntimeError(f"Failed to generate valid brief for {subscriber_id} after 3 attempts.")
        
    return {
        "subscriber_id": subscriber_id,
        "tenure_months": tenure_months,
        "failed_recharges_6m": failed_recharges_6m,
        "churn_probability": round(churn_prob, 4),
        "risk_band": risk_band,
        "segment_name": segment_name,
        "recommended_action": validated_brief.recommended_action,
        "policy_source": validated_brief.policy_source
    }

if __name__ == "__main__":
    prio_df = pd.read_csv(PRIORITY_PATH)
    mock_llm.reset()
    results = []
    print(f"Executing Retention Copilot for {len(prio_df)} subscribers in priority list...")
    for sid in prio_df['subscriber_id']:
        b = build_brief(sid)
        results.append(b)
        print(f"  {b['subscriber_id']} | Risk: {b['risk_band']:6s} | Action: {b['recommended_action']}")
        
    out_file = os.path.join(OUTPUT_DIR, "retention_briefs.jsonl")
    with open(out_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nWrote {len(results)} validated retention briefs to {out_file}")
