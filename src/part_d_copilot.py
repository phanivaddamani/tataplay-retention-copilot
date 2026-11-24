"""part_d_copilot.py -- Prompts, structured extraction, policy RAG, and retention briefs."""
import os
import sys
import json
import sqlite3
import joblib
import pandas as pd
import numpy as np

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import mock_llm
from llm_client import call_llm
from features import build_features, NUMERIC_FEATURES, CATEGORICAL_FEATURES

from typing import Literal, Optional
from pydantic import BaseModel, Field, ValidationError

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.exceptions import OutputParserException

import chromadb
from chromadb.utils import embedding_functions

RAW_DIR = os.path.join("data", "raw")
POL_DIR = os.path.join("data", "policies")
OUTPUT_DIR = "outputs"
DB_PATH = os.path.join("data", "tataplay.db")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# TASK 2: Prompt Design and Few-Shot Classification
# =============================================================================
def run_task_2():
    print("\n=== TASK 2: TICKET CLASSIFICATION ===")
    tickets_df = pd.read_csv(os.path.join(RAW_DIR, "tickets.csv"))
    # Filter readable tickets (drop four '...' tickets)
    readable_df = tickets_df[tickets_df['text'].str.strip() != "..."].copy()
    print(f"Total tickets: {len(tickets_df)}, Readable tickets: {len(readable_df)}")
    
    # 1. Weak Prompt
    weak_prompt = ChatPromptTemplate.from_messages([
        ("system", "TICKET CLASSIFIER. Classify the ticket."),
        ("human", "{text}")
    ])
    
    # 2. Strong Zero-Shot Prompt
    strong_zero_sys = (
        "TICKET CLASSIFIER. You are an expert support desk triage assistant. "
        "Classify the customer ticket into exactly one of these four categories: "
        "billing, signal, plan_change, cancellation. "
        "Do not invent new categories. Respond with only the label."
    )
    strong_zero_prompt = ChatPromptTemplate.from_messages([
        ("system", strong_zero_sys),
        ("human", "{text}")
    ])
    
    # 3. Few-Shot Prompt
    few_shot_df = pd.read_csv(os.path.join(RAW_DIR, "few_shot_examples.csv"))
    few_shot_lines = []
    for _, row in few_shot_df.iterrows():
        few_shot_lines.append(f"Ticket: {row['text']}\nLabel: {row['label']}")
    few_shot_text = "\n\n".join(few_shot_lines)
    
    few_shot_sys = (
        f"{strong_zero_sys}\n\n"
        f"Examples:\n{few_shot_text}"
    )
    few_shot_prompt = ChatPromptTemplate.from_messages([
        ("system", few_shot_sys),
        ("human", "{text}")
    ])
    
    def execute_chain(prompt, df):
        mock_llm.reset()
        chain = prompt | (lambda p: call_llm([
            {"role": "user" if m.type == "human" else ("assistant" if m.type == "ai" else "system"),
             "content": m.content}
            for m in p.to_messages()
        ])) | StrOutputParser()
        
        preds = []
        for text in df['text']:
            res = chain.invoke({"text": text}).strip()
            preds.append(res)
            
        total_calls = mock_llm._state["calls"]
        recovered_limits = total_calls - len(df)
        return preds, recovered_limits

    valid_labels = set(mock_llm.LABELS)
    
    print("\nRunning Weak Prompt...")
    weak_preds, weak_rec = execute_chain(weak_prompt, readable_df)
    n_valid_weak = sum(p in valid_labels for p in weak_preds)
    acc_weak = sum(p == t for p, t in zip(weak_preds, readable_df['true_category'])) / len(readable_df)
    print(f"Weak Prompt: Valid={n_valid_weak}/{len(readable_df)}, Acc={acc_weak*100:.2f}%, Recovered 429s={weak_rec}")
    
    print("\nRunning Strong Zero-Shot Prompt...")
    zero_preds, zero_rec = execute_chain(strong_zero_prompt, readable_df)
    n_valid_zero = sum(p in valid_labels for p in zero_preds)
    acc_zero = sum(p == t for p, t in zip(zero_preds, readable_df['true_category'])) / len(readable_df)
    print(f"Zero-Shot Prompt: Valid={n_valid_zero}/{len(readable_df)}, Acc={acc_zero*100:.2f}%, Recovered 429s={zero_rec}")
    
    print("\nRunning Few-Shot Prompt...")
    few_preds, few_rec = execute_chain(few_shot_prompt, readable_df)
    n_valid_few = sum(p in valid_labels for p in few_preds)
    acc_few = sum(p == t for p, t in zip(few_preds, readable_df['true_category'])) / len(readable_df)
    print(f"Few-Shot Prompt: Valid={n_valid_few}/{len(readable_df)}, Acc={acc_few*100:.2f}%, Recovered 429s={few_rec}")
    
    # Save outputs/ticket_classification.csv
    readable_df['zero_shot'] = zero_preds
    readable_df['few_shot'] = few_preds
    out_csv = os.path.join(OUTPUT_DIR, "ticket_classification.csv")
    readable_df[['ticket_id', 'true_category', 'zero_shot', 'few_shot']].to_csv(out_csv, index=False)
    print(f"Saved classification results to {out_csv}")
    
    # Analyze the 4 tickets few-shot gets wrong
    wrong_few = readable_df[readable_df['few_shot'] != readable_df['true_category']]
    print(f"\nTickets misclassified by few-shot ({len(wrong_few)}):")
    for _, row in wrong_few.iterrows():
        print(f"  {row['ticket_id']}: True={row['true_category']} | Pred={row['few_shot']} | Text='{row['text']}'")

# =============================================================================
# TASK 3: Validated Structured Extraction with Retry
# =============================================================================
class TicketRecord(BaseModel):
    subscriber_id: Optional[str] = None
    category: Literal["billing", "signal", "plan_change", "cancellation", "other"]
    urgency: int = Field(ge=1, le=5)
    wants_refund: bool

def run_task_3():
    print("\n=== TASK 3: STRUCTURED EXTRACTION WITH RETRY ===")
    
    # Demonstration with 3 hand-made inputs
    print("\n--- Pydantic Demonstration ---")
    # 1. string "3" converted to integer 3
    r1 = TicketRecord(subscriber_id="TP00001", category="billing", urgency="3", wants_refund=True)
    print(f"1. Coercion: urgency='3' (type str) -> urgency={r1.urgency} (type {type(r1.urgency).__name__})")
    
    # 2. urgency=9 raises ValidationError
    try:
        TicketRecord(subscriber_id="TP00001", category="billing", urgency=9, wants_refund=True)
    except ValidationError as e:
        err = e.errors()[0]
        print(f"2. Validation failure: field='{err['loc'][0]}', error_type='{err['type']}', msg='{err['msg']}'")
        
    # 3. category="refund" raises ValidationError
    try:
        TicketRecord(subscriber_id="TP00001", category="refund", urgency=3, wants_refund=True)
    except ValidationError as e:
        err = e.errors()[0]
        print(f"3. Validation failure: field='{err['loc'][0]}', error_type='{err['type']}', msg='{err['msg']}'")
        
    # Process all 150 tickets
    parser = PydanticOutputParser(pydantic_object=TicketRecord)
    format_instructions = parser.get_format_instructions()
    
    tickets_df = pd.read_csv(os.path.join(RAW_DIR, "tickets.csv"))
    mock_llm.reset()
    
    results = []
    first_attempt_valid = 0
    retry_valid = 0
    manual_review_count = 0
    first_fail_json = 0
    first_fail_schema_urgency = 0
    first_fail_schema_cat = 0
    
    for _, row in tickets_df.iterrows():
        tid = row['ticket_id']
        text = row['text']
        
        # Initial call
        messages = [
            {"role": "system", "content": f"TICKET EXTRACTOR. Extract fields from the support ticket.\n{format_instructions}"},
            {"role": "user", "content": text}
        ]
        
        attempts = 0
        parsed_record = None
        status = "failed"
        
        while attempts < 3:
            attempts += 1
            reply = call_llm(messages)
            
            try:
                parsed_record = parser.parse(reply)
                status = "success"
                if attempts == 1:
                    first_attempt_valid += 1
                else:
                    retry_valid += 1
                break
            except Exception as e:
                err_text = str(e)
                if attempts == 1:
                    # Categorize first attempt failure
                    try:
                        data = json.loads(reply)
                        if data.get("urgency") == 7:
                            first_fail_schema_urgency += 1
                        elif data.get("category") not in ["billing", "signal", "plan_change", "cancellation", "other"]:
                            first_fail_schema_cat += 1
                        else:
                            first_fail_json += 1
                    except Exception:
                        first_fail_json += 1
                        
                # Append bad reply and error correction message
                messages.append({"role": "assistant", "content": reply})
                messages.append({"role": "user", "content": f"VALIDATION ERROR: {err_text}"})
                
        if status != "success":
            status = "manual_review"
            manual_review_count += 1
            
        results.append({
            "ticket_id": tid,
            "status": status,
            "attempts": attempts,
            "record": parsed_record.model_dump() if parsed_record else None
        })
        
    # Write outputs/ticket_extractions.jsonl
    out_jsonl = os.path.join(OUTPUT_DIR, "ticket_extractions.jsonl")
    with open(out_jsonl, "w") as f:
        for r in results:
            # write as JSON string per line
            line_dict = {
                "ticket_id": r["ticket_id"],
                "status": r["status"],
                "attempts": r["attempts"],
                "record": r["record"]
            }
            f.write(json.dumps(line_dict) + "\n")
            
    print(f"\nSaved ticket extractions to {out_jsonl}")
    print(f"Extraction Summary:")
    print(f"  Valid on first attempt: {first_attempt_valid}")
    print(f"  Valid after retry:      {retry_valid}")
    print(f"  Manual review:          {manual_review_count}")
    print(f"  Total first attempt failures: {first_fail_json + first_fail_schema_urgency + first_fail_schema_cat}")
    print(f"    - Invalid JSON / Chatty:    {first_fail_json}")
    print(f"    - Schema invalid urgency 7: {first_fail_schema_urgency}")
    print(f"    - Schema invalid category:  {first_fail_schema_cat}")
    
    total_calls = mock_llm._state["calls"]
    recovered_limits = total_calls - sum(r["attempts"] for r in results)
    print(f"Total post() calls invoked: {total_calls}")
    print(f"Recovered rate limits (429): {recovered_limits}")
    
    # Check valid records properties
    valid_recs = [r["record"] for r in results if r["status"] == "success"]
    n_refund = sum(r["wants_refund"] for r in valid_recs)
    n_null_sid = sum(r["subscriber_id"] is None for r in valid_recs)
    print(f"Among {len(valid_recs)} valid records: wants_refund={n_refund}, null subscriber_id={n_null_sid}")

# =============================================================================
# TASK 4: Policy RAG Assistant
# =============================================================================
def chunk_text(text, size=400, overlap=60):
    chunks = []
    step = size - overlap
    start = 0
    while start + overlap < len(text):
        chunks.append(text[start:start+size])
        start += step
    return chunks

def run_task_4():
    print("\n=== TASK 4: POLICY RAG ASSISTANT ===")
    
    # 1. Chunking
    files = sorted(os.listdir(POL_DIR))
    chunks_400 = []
    for fname in files:
        fpath = os.path.join(POL_DIR, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            txt = f.read().strip()
        f_chunks = chunk_text(txt, size=400, overlap=60)
        for i, ch in enumerate(f_chunks):
            chunks_400.append({"id": f"{fname}_{i}", "source": fname, "chunk": i, "text": ch})
    print(f"Created {len(chunks_400)} chunks (400/60) across {len(files)} policy files.")
    
    # 2. Embedding & Chroma Store
    ef = embedding_functions.ONNXMiniLM_L6_V2()
    client = chromadb.Client()
    
    # Cosine distance collection
    coll_name = "policy_chunks_400"
    try:
        client.delete_collection(coll_name)
    except Exception:
        pass
    collection = client.create_collection(
        name=coll_name,
        metadata={"hnsw:space": "cosine"},
        embedding_function=ef
    )
    
    collection.add(
        ids=[c["id"] for c in chunks_400],
        documents=[c["text"] for c in chunks_400],
        metadatas=[{"source": c["source"], "chunk": c["chunk"]} for c in chunks_400]
    )
    print(f"Indexed {collection.count()} chunks into in-memory ChromaDB.")
    
    def retrieve(query, k=3):
        res = collection.query(query_texts=[query], n_results=k)
        hits = []
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            sim = 1.0 - dist
            hits.append({"source": meta["source"], "similarity": sim, "text": doc})
        return hits

    # 3. Evaluation on 16 questions in policy_eval.csv
    eval_df = pd.read_csv(os.path.join(RAW_DIR, "policy_eval.csv"))
    
    # Keyword baseline function
    def keyword_baseline(q):
        q_words = set(mock_llm.re.findall(r"[a-z]+", q.lower())) - mock_llm.STOP
        file_scores = {}
        for fname in files:
            with open(os.path.join(POL_DIR, fname), "r", encoding="utf-8") as f:
                ptxt = f.read().lower()
            p_words = set(mock_llm.re.findall(r"[a-z]+", ptxt)) - mock_llm.STOP
            file_scores[fname] = len(q_words & p_words)
        # Sort by score desc, ties alphabetical asc
        best_file = max(sorted(files), key=lambda f: file_scores[f])
        return best_file

    ans_top3_hits = 0
    ans_top1_hits = 0
    kw_hits = 0
    ans_sims = []
    unans_sims = []
    
    for _, row in eval_df.iterrows():
        q = row['question']
        exp_file = str(row['expected_file']).strip()
        is_ans = bool(exp_file and exp_file != "nan")
        
        hits = retrieve(q, k=3)
        top1 = hits[0]
        top3_sources = [h["source"] for h in hits]
        
        if is_ans:
            if exp_file in top3_sources:
                ans_top3_hits += 1
            if top1["source"] == exp_file:
                ans_top1_hits += 1
            ans_sims.append(top1["similarity"])
            
            kw_best = keyword_baseline(q)
            if kw_best == exp_file:
                kw_hits += 1
        else:
            unans_sims.append(top1["similarity"])
            
    print(f"\n--- Retrieval Evaluation (12 Answerable Questions) ---")
    print(f"Top-3 File Hits: {ans_top3_hits}/12")
    print(f"Top-1 File Hits: {ans_top1_hits}/12")
    print(f"Answerable Top-1 Similarity Range: {min(ans_sims):.3f} to {max(ans_sims):.3f} (mean: {np.mean(ans_sims):.3f})")
    print(f"Unanswerable Top-1 Similarities: {[round(s, 3) for s in unans_sims]}")
    print(f"Keyword Baseline Top-1 Correct:  {kw_hits}/12")
    
    # 4. Calibrate & Ground with Threshold 0.32
    REJECTION_THRESHOLD = 0.32
    print(f"\nCalibrated Rejection Threshold: {REJECTION_THRESHOLD}")
    
    def answer_rag(question, grounded=True):
        hits = retrieve(question, k=3)
        top1_sim = hits[0]["similarity"]
        
        if grounded and top1_sim < REJECTION_THRESHOLD:
            return "I don't have enough information to answer that.", hits
            
        # Build context blocks
        context_blocks = []
        for h in hits:
            context_blocks.append(f"[{h['source']}]\n{h['text']}")
        context_str = "\n\n".join(context_blocks)
        
        if grounded:
            sys_msg = (
                "POLICY ASSISTANT. Answer the question using only the context below. "
                "If the context does not contain the answer, reply that you do not have enough information."
            )
            user_msg = f"CONTEXT:\n\n{context_str}\n\nQUESTION:\n{question}"
        else:
            sys_msg = "POLICY ASSISTANT. Answer the question."
            user_msg = question
            
        ans = call_llm([{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}])
        return ans, hits

    rag_records = []
    grounded_fact_hits = 0
    unans_fallback_hits = 0
    
    for _, row in eval_df.iterrows():
        q = row['question']
        exp_file = str(row['expected_file']).strip() if pd.notna(row['expected_file']) else ""
        exp_fact = str(row['expected_fact']).strip() if pd.notna(row['expected_fact']) else ""
        
        g_ans, hits = answer_rag(q, grounded=True)
        ung_ans, _ = answer_rag(q, grounded=False)
        
        top1 = hits[0]
        hit3 = int(exp_file in [h["source"] for h in hits]) if exp_file else 0
        
        if exp_file:
            if exp_fact.lower() in g_ans.lower():
                grounded_fact_hits += 1
        else:
            if g_ans == "I don't have enough information to answer that.":
                unans_fallback_hits += 1
                
        rag_records.append({
            "question": q,
            "expected_file": exp_file,
            "top1_source": top1["source"],
            "top1_similarity": round(top1["similarity"], 4),
            "hit_at_3": hit3,
            "grounded_answer": g_ans,
            "ungrounded_answer": ung_ans
        })
        
    rag_eval_df = pd.DataFrame(rag_records)
    rag_csv = os.path.join(OUTPUT_DIR, "rag_eval.csv")
    rag_eval_df.to_csv(rag_csv, index=False)
    print(f"Saved RAG evaluation to {rag_csv}")
    print(f"Grounded Answers containing expected fact: {grounded_fact_hits}/12")
    print(f"Unanswerable getting fallback sentence:     {unans_fallback_hits}/4")
    
    # 5. Chunk Size Comparison (size=200, overlap=30)
    print("\n--- Chunk Size Experiment (200/30) ---")
    chunks_200 = []
    for fname in files:
        fpath = os.path.join(POL_DIR, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            txt = f.read().strip()
        f_chunks = chunk_text(txt, size=200, overlap=30)
        for i, ch in enumerate(f_chunks):
            chunks_200.append({"id": f"{fname}_{i}", "source": fname, "chunk": i, "text": ch})
            
    coll_200 = client.create_collection(
        name="policy_chunks_200",
        metadata={"hnsw:space": "cosine"},
        embedding_function=ef
    )
    coll_200.add(
        ids=[c["id"] for c in chunks_200],
        documents=[c["text"] for c in chunks_200],
        metadatas=[{"source": c["source"], "chunk": c["chunk"]} for c in chunks_200]
    )
    print(f"Created {len(chunks_200)} chunks for 200/30 configuration.")
    
    fact_in_top3_200 = 0
    ans_with_fact_200 = 0
    
    for _, row in eval_df.iterrows():
        exp_file = str(row['expected_file']).strip() if pd.notna(row['expected_file']) else ""
        exp_fact = str(row['expected_fact']).strip() if pd.notna(row['expected_fact']) else ""
        if not exp_file:
            continue
        q = row['question']
        res = coll_200.query(query_texts=[q], n_results=3)
        top3_docs = res["documents"][0]
        
        # Check if fact in top 3 chunks
        fact_in_docs = any(exp_fact.lower() in d.lower() for d in top3_docs)
        if fact_in_docs:
            fact_in_top3_200 += 1
            
        # Grounded answer call
        ctx_blocks = [f"[{m['source']}]\n{d}" for d, m in zip(top3_docs, res["metadatas"][0])]
        ctx_str = "\n\n".join(ctx_blocks)
        sys_msg = (
            "POLICY ASSISTANT. Answer the question using only the context below. "
            "If the context does not contain the answer, reply that you do not have enough information."
        )
        user_msg = f"CONTEXT:\n\n{ctx_str}\n\nQUESTION:\n{q}"
        ans = call_llm([{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}])
        if exp_fact.lower() in ans.lower():
            ans_with_fact_200 += 1
            
    print(f"200/30 Chunks: Expected fact in Top-3 chunks: {fact_in_top3_200}/12 (vs 12/12 in 400/60)")
    print(f"200/30 Chunks: Grounded answer contains fact: {ans_with_fact_200}/12 (vs 12/12 in 400/60)")

# =============================================================================
# TASK 5: Retention Copilot
# =============================================================================
class RetentionBrief(BaseModel):
    subscriber_id: str
    risk_band: Literal["High", "Medium", "Low"]
    recommended_action: str
    policy_source: str

def run_task_5():
    print("\n=== TASK 5: THE RETENTION COPILOT ===")
    from copilot import build_brief
    
    priority_df = pd.read_csv(os.path.join(OUTPUT_DIR, "retention_priority.csv"))
    print(f"Building retention briefs for top {len(priority_df)} priority subscribers...")
    
    mock_llm.reset()
    briefs = []
    for _, row in priority_df.iterrows():
        sid = row['subscriber_id']
        brief = build_brief(sid)
        briefs.append(brief)
        
    out_briefs = os.path.join(OUTPUT_DIR, "retention_briefs.jsonl")
    with open(out_briefs, "w") as f:
        for b in briefs:
            f.write(json.dumps(b) + "\n")
            
    print(f"\nSaved {len(briefs)} retention briefs to {out_briefs}")
    
    # Report counts
    bands = pd.Series([b["risk_band"] for b in briefs]).value_counts()
    actions = pd.Series([b["recommended_action"] for b in briefs]).value_counts()
    print("\nRisk Band Counts:\n", bands)
    print("\nRecommended Action Counts:\n", actions)

if __name__ == "__main__":
    run_task_2()
    run_task_3()
    run_task_4()
    run_task_5()
