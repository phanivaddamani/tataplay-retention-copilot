-- Q1_revenue_by_plan
SELECT 
    p.plan_name,
    SUM(r.amount_inr) AS total_revenue,
    ROUND(AVG(r.amount_inr), 2) AS avg_recharge,
    COUNT(DISTINCT s.subscriber_id) AS distinct_subscribers
FROM recharges r
INNER JOIN subscribers s ON r.subscriber_id = s.subscriber_id
INNER JOIN plans p ON s.plan_id = p.plan_id
WHERE r.status = 'success'
GROUP BY p.plan_name
ORDER BY total_revenue DESC;

-- Q2_never_successfully_recharged
SELECT 
    s.subscriber_id,
    s.tenure_months,
    s.monthly_recharge_inr
FROM subscribers s
LEFT JOIN recharges r 
    ON s.subscriber_id = r.subscriber_id 
    AND r.status = 'success'
WHERE r.recharge_id IS NULL;

-- Q3_ranking_within_city
WITH subscriber_spend AS (
    SELECT 
        s.subscriber_id,
        s.city,
        SUM(r.amount_inr) AS total_spend
    FROM subscribers s
    INNER JOIN recharges r ON s.subscriber_id = r.subscriber_id
    WHERE r.status = 'success'
    GROUP BY s.subscriber_id, s.city
),
ranked_spend AS (
    SELECT 
        subscriber_id,
        city,
        total_spend,
        RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS rnk,
        DENSE_RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS dense_rnk,
        ROW_NUMBER() OVER (PARTITION BY city ORDER BY total_spend DESC, subscriber_id ASC) AS row_num
    FROM subscriber_spend
)
SELECT * FROM ranked_spend;

-- Q4_running_total
SELECT 
    recharge_id,
    recharge_date,
    amount_inr,
    SUM(amount_inr) OVER (ORDER BY recharge_date ASC, recharge_id ASC) AS running_total
FROM recharges
WHERE subscriber_id = 'TP00010' AND status = 'success'
ORDER BY recharge_date ASC, recharge_id ASC;

