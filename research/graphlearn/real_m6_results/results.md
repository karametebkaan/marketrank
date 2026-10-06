# M6 results

data gate: `{"sessions": 503, "median_nodes": 964.0, "tickers": 1000, "bars": 13018, "passes": true}`

**PASS: False**

| run | variant | IC | - B0E | - Bshuf | - B0 |
|---|---|---|---|---|---|
| scratch_s1 | Bprior | +0.0083 (t=+3.0) | +0.0009 (t=+0.4, months+ 35%) | -0.0002 (t=-0.1, months+ 48%) | +0.0006 (t=+0.6, months+ 39%) |
| scratch_s1 | Bshuf | +0.0085 (t=+3.1) | +0.0011 (t=+0.4, months+ 39%) |  | +0.0008 (t=+0.7, months+ 43%) |
| scratch_s1 | B0E | +0.0074 (t=+2.9) |  | -0.0011 (t=-0.4, months+ 61%) | -0.0004 (t=-0.1, months+ 65%) |
| scratch_s1 | B0 | +0.0078 (t=+2.9) | +0.0004 (t=+0.1, months+ 35%) | -0.0008 (t=-0.7, months+ 57%) |  |
| rolling_s1 | Bprior | +0.0089 (t=+2.7) | -0.0014 (t=-0.5, months+ 39%) | +0.0001 (t=+0.2, months+ 70%) | +0.0005 (t=+0.6, months+ 65%) |
| rolling_s1 | Bshuf | +0.0088 (t=+2.7) | -0.0014 (t=-0.5, months+ 30%) |  | +0.0004 (t=+0.7, months+ 65%) |
| rolling_s1 | B0E | +0.0102 (t=+3.5) |  | +0.0014 (t=+0.5, months+ 70%) | +0.0018 (t=+0.7, months+ 61%) |
| rolling_s1 | B0 | +0.0084 (t=+2.6) | -0.0018 (t=-0.7, months+ 39%) | -0.0004 (t=-0.7, months+ 35%) |  |
| rolling_s2 | Bprior | +0.0092 (t=+3.4) | -0.0011 (t=-0.5, months+ 39%) | +0.0006 (t=+0.9, months+ 61%) | +0.0018 (t=+1.4, months+ 57%) |
| rolling_s2 | Bshuf | +0.0086 (t=+3.1) | -0.0018 (t=-0.7, months+ 48%) |  | +0.0011 (t=+0.9, months+ 57%) |
| rolling_s2 | B0E | +0.0103 (t=+3.8) |  | +0.0018 (t=+0.7, months+ 52%) | +0.0029 (t=+1.0, months+ 74%) |
| rolling_s2 | B0 | +0.0074 (t=+2.5) | -0.0029 (t=-1.0, months+ 26%) | -0.0011 (t=-0.9, months+ 43%) |  |

Decile spread of label_6 (non-overlapping bars; net of 20 bps per period):

| run | variant | gross | t | net |
|---|---|---|---|---|
| scratch_s1 | Bprior | -0.00053 | -2.8 | -0.00253 |
| scratch_s1 | Bshuf | -0.00041 | -2.3 | -0.00241 |
| scratch_s1 | B0E | -0.00027 | -2.1 | -0.00227 |
| scratch_s1 | B0 | -0.00051 | -2.6 | -0.00251 |
| rolling_s1 | Bprior | -0.00041 | -2.3 | -0.00241 |
| rolling_s1 | Bshuf | -0.00043 | -2.4 | -0.00243 |
| rolling_s1 | B0E | -0.00015 | -1.3 | -0.00215 |
| rolling_s1 | B0 | -0.00036 | -2.0 | -0.00236 |
| rolling_s2 | Bprior | -0.00051 | -2.6 | -0.00251 |
| rolling_s2 | Bshuf | -0.00050 | -2.6 | -0.00250 |
| rolling_s2 | B0E | -0.00022 | -1.7 | -0.00222 |
| rolling_s2 | B0 | -0.00056 | -2.9 | -0.00256 |
