# Eval run 2026-11-19 23:21

## Summary

| mode | recall@3 | answer accuracy |
|---|---|---|
| vector | 92% | 92% |
| hybrid | 92% | 92% |
| rerank | 100% | 100% |

## vector

| id | question | retrieved (rank, top 3) | answer | missing |
|---|---|---|---|---|
| expense-01 | How soon must I submit an expense report? | PASS (1) | PASS | - |
| expense-02 | What is the mileage rate for my own car? | PASS (1) | PASS | - |
| expense-03 | If I lose a receipt, which form do I fill in and who has to sign it? | PASS (1) | PASS | - |
| expense-04 | Who signs EX-15? | FAIL (-) | FAIL | approver |
| remote-01 | How much is the home office equipment stipend? | PASS (1) | PASS | - |
| remote-02 | How many days a week must hybrid employees be in the office? | PASS (1) | PASS | - |
| remote-03 | What is Form RW-7 used for? | PASS (2) | PASS | - |
| security-01 | How long must my password be? | PASS (1) | PASS | - |
| security-02 | How quickly must I report a security incident? | PASS (1) | PASS | - |
| security-03 | How long after a customer's contract ends can we keep their data? | PASS (1) | PASS | - |
| pto-01 | How many PTO days do employees get per year? | PASS (1) | PASS | - |
| pto-02 | How many unused PTO days can I carry over into next year? | PASS (2) | PASS | - |
| none-01 | What is the CEO's favourite colour? | n/a | PASS | - |

recall@3: 92% (11/12)
answer accuracy: 92% (12/13)

## hybrid

| id | question | retrieved (rank, top 3) | answer | missing |
|---|---|---|---|---|
| expense-01 | How soon must I submit an expense report? | PASS (1) | PASS | - |
| expense-02 | What is the mileage rate for my own car? | FAIL (-) | FAIL | $0.67 |
| expense-03 | If I lose a receipt, which form do I fill in and who has to sign it? | PASS (1) | PASS | - |
| expense-04 | Who signs EX-15? | PASS (1) | PASS | - |
| remote-01 | How much is the home office equipment stipend? | PASS (1) | PASS | - |
| remote-02 | How many days a week must hybrid employees be in the office? | PASS (1) | PASS | - |
| remote-03 | What is Form RW-7 used for? | PASS (1) | PASS | - |
| security-01 | How long must my password be? | PASS (1) | PASS | - |
| security-02 | How quickly must I report a security incident? | PASS (1) | PASS | - |
| security-03 | How long after a customer's contract ends can we keep their data? | PASS (1) | PASS | - |
| pto-01 | How many PTO days do employees get per year? | PASS (3) | PASS | - |
| pto-02 | How many unused PTO days can I carry over into next year? | PASS (2) | PASS | - |
| none-01 | What is the CEO's favourite colour? | n/a | PASS | - |

recall@3: 92% (11/12)
answer accuracy: 92% (12/13)

## rerank

| id | question | retrieved (rank, top 3) | answer | missing |
|---|---|---|---|---|
| expense-01 | How soon must I submit an expense report? | PASS (1) | PASS | - |
| expense-02 | What is the mileage rate for my own car? | PASS (1) | PASS | - |
| expense-03 | If I lose a receipt, which form do I fill in and who has to sign it? | PASS (1) | PASS | - |
| expense-04 | Who signs EX-15? | PASS (1) | PASS | - |
| remote-01 | How much is the home office equipment stipend? | PASS (1) | PASS | - |
| remote-02 | How many days a week must hybrid employees be in the office? | PASS (1) | PASS | - |
| remote-03 | What is Form RW-7 used for? | PASS (1) | PASS | - |
| security-01 | How long must my password be? | PASS (1) | PASS | - |
| security-02 | How quickly must I report a security incident? | PASS (1) | PASS | - |
| security-03 | How long after a customer's contract ends can we keep their data? | PASS (1) | PASS | - |
| pto-01 | How many PTO days do employees get per year? | PASS (2) | PASS | - |
| pto-02 | How many unused PTO days can I carry over into next year? | PASS (2) | PASS | - |
| none-01 | What is the CEO's favourite colour? | n/a | PASS | - |

recall@3: 100% (12/12)
answer accuracy: 100% (13/13)

---

