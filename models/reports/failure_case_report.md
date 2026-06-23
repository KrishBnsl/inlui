# Failure Case Analysis

Analyzing the top 100 prediction errors to identify systemic patterns.

### Common Institute Types in Failures
| institute_type_str   |   count |
|:---------------------|--------:|
| NIT                  |     100 |

### Common Categories in Failures
| category_str   |   count |
|:---------------|--------:|
| OPEN           |      95 |
| OBC-NCL        |       5 |

### Common Rounds in Failures
|   round |   count |
|--------:|--------:|
|       1 |      33 |
|       5 |      14 |
|       6 |      14 |
|       4 |      14 |
|       2 |      13 |

### Top 10 Specific Failures
| institute                                 | program                                   | category_str   |   round |   actual_closing_rank |   predicted_closing_rank |   residual |
|:------------------------------------------|:------------------------------------------|:---------------|--------:|----------------------:|-------------------------:|-----------:|
| National Institute of Technology, Mizoram | Electronics and Communication Engineering | OPEN           |       1 |           1.49704e+06 |                  1496957 |    81.3827 |
| National Institute of Technology, Mizoram | Chemical Science and Technology           | OPEN           |       1 |           1.03295e+06 |                  1032885 |    65.8291 |
| National Institute of Technology, Mizoram | Chemical Science and Technology           | OPEN           |       2 |           1.27491e+06 |                  1274845 |    65.1298 |
| National Institute of Technology, Mizoram | Mathematics and Computing                 | OPEN           |       1 |      936083           |                   936022 |    61.3396 |
| National Institute of Technology, Mizoram | Mathematics and Computing                 | OPEN           |       1 |           1.01934e+06 |                  1019286 |    58.4986 |
| National Institute of Technology, Mizoram | Electrical Engineering                    | OPEN           |       1 |           1.22172e+06 |                  1221660 |    57.2709 |
| National Institute of Technology, Mizoram | Chemical Science and Technology           | OPEN           |       1 |      937704           |                   937648 |    56.4048 |
| National Institute of Technology, Mizoram | Mathematics and Computing                 | OPEN           |       2 |           1.2108e+06  |                  1210742 |    55.4647 |
| National Institute of Technology, Mizoram | Electrical Engineering                    | OPEN           |       2 |           1.17673e+06 |                  1176674 |    52.525  |
| National Institute of Technology, Mizoram | Electrical Engineering                    | OPEN           |       1 |      778631           |                   778580 |    51.4855 |

### Root Causes & Future Work
- High-volatility branches or sudden popularity shifts in newer IITs/NITs.
- Specific categories (e.g. PwD) may have very low seat counts leading to erratic cutoffs.