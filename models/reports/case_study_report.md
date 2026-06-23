# JoSAA Day 4: Recommendation Engine Case Studies

This report demonstrates the uncertainty-aware recommendation engine output for three distinct student profiles. The engine utilizes Monte Carlo simulation (N=1000) over the Day 3 Ridge regression model to provide probabilistic admission estimates rather than fragile point predictions.

## Profile: High Achiever
- **Student Rank:** 2000
- **Category:** OPEN
- **Preferences:** IIT - Computer Science

### Top 10 Recommended Options
| Institute | Program | Predicted Cutoff | Admission Prob | 90% CI Bounds | Classification |
|---|---|---|---|---|---|
| Indian Institute of Technology Bhilai | Computer Science and Engineering | 7382 | 100.0% | [7224 - 7547] | Safe |
| Indian Institute of Technology Dharwad | Computer Science and Engineering | 7208 | 100.0% | [7052 - 7380] | Safe |
| Indian Institute of Technology Jammu | Computer Science and Engineering | 6651 | 100.0% | [6487 - 6816] | Safe |
| Indian Institute of Technology Palakkad | Computer Science and Engineering | 6454 | 100.0% | [6287 - 6626] | Safe |
| Indian Institute of Technology Goa | Computer Science and Engineering | 6242 | 100.0% | [6077 - 6414] | Safe |
| Indian Institute of Technology Tirupati | Computer Science and Engineering | 5034 | 100.0% | [4867 - 5202] | Safe |
| Indian Institute of Technology Patna | B.Tech (Computer Science and Engineering) - MBA in Digital Business Management (IIM Bodh Gaya) | 4297 | 100.0% | [4145 - 4476] | Safe |
| Indian Institute of Technology Bhubaneswar | Computer Science and Engineering | 4162 | 100.0% | [3995 - 4329] | Safe |
| Indian Institute of Technology (ISM) Dhanbad | Computer Science and Engineering | 3585 | 100.0% | [3413 - 3747] | Safe |
| Indian Institute of Technology Patna | Computer Science and Engineering | 3377 | 100.0% | [3214 - 3543] | Safe |



## Profile: Mid-Tier Aspirant
- **Student Rank:** 15000
- **Category:** OBC-NCL
- **Preferences:** NIT - Electronics

### Top 10 Recommended Options
| Institute | Program | Predicted Cutoff | Admission Prob | 90% CI Bounds | Classification |
|---|---|---|---|---|---|
| National Institute of Technology Sikkim | Electrical and Electronics Engineering | 476705 | 100.0% | [476541 - 476863] | Safe |
| National Institute of Technology, Manipur | Electronics and Communication Engineering | 196250 | 100.0% | [196086 - 196420] | Safe |
| National Institute of Technology Sikkim | Electronics and Communication Engineering | 172825 | 100.0% | [172662 - 172990] | Safe |
| National Institute of Technology Goa | Electrical and Electronics Engineering | 149436 | 100.0% | [149281 - 149590] | Safe |
| National Institute of Technology, Srinagar | Electronics and Communication Engineering | 145774 | 100.0% | [145602 - 145931] | Safe |
| National Institute of Technology Goa | Electronics and Communication Engineering | 94812 | 100.0% | [94641 - 94983] | Safe |
| National Institute of Technology Hamirpur | Electronics and Communication Engineering | 50316 | 100.0% | [50152 - 50473] | Safe |
| National Institute of Technology Puducherry | Electrical and Electronics Engineering | 48451 | 100.0% | [48272 - 48614] | Safe |
| National Institute of Technology Arunachal Pradesh | Electronics and Communication Engineering | 40278 | 100.0% | [40123 - 40435] | Safe |
| National Institute of Technology Puducherry | Electronics and Communication Engineering | 36972 | 100.0% | [36795 - 37135] | Safe |



## Profile: Targeted State Student
- **Student Rank:** 40000
- **Category:** OPEN
- **Preferences:** NIT - Civil

### Top 10 Recommended Options
| Institute | Program | Predicted Cutoff | Admission Prob | 90% CI Bounds | Classification |
|---|---|---|---|---|---|
| National Institute of Technology, Mizoram | Civil Engineering | 626229 | 100.0% | [626062 - 626398] | Safe |
| National Institute of Technology Sikkim | Civil Engineering | 366110 | 100.0% | [365941 - 366264] | Safe |
| National Institute of Technology, Manipur | Civil Engineering | 287838 | 100.0% | [287667 - 288008] | Safe |
| National Institute of Technology Meghalaya | Civil Engineering | 259471 | 100.0% | [259312 - 259630] | Safe |
| National Institute of Technology Agartala | Civil Engineering | 237333 | 100.0% | [237163 - 237500] | Safe |
| National Institute of Technology Goa | Civil Engineering | 178345 | 100.0% | [178174 - 178501] | Safe |
| National Institute of Technology Goa | Civil Engineering | 168173 | 100.0% | [168028 - 168338] | Safe |
| National Institute of Technology Puducherry | Civil Engineering | 142535 | 100.0% | [142372 - 142709] | Safe |
| National Institute of Technology, Srinagar | Civil Engineering | 135449 | 100.0% | [135283 - 135606] | Safe |
| National Institute of Technology Hamirpur | Civil Engineering | 111968 | 100.0% | [111809 - 112126] | Safe |


