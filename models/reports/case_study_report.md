# JoSAA Day 4: Recommendation Engine Case Studies

This report demonstrates the uncertainty-aware recommendation engine output for three distinct student profiles. The engine utilizes Monte Carlo simulation (N=1000) over the Day 3 Ridge regression model to provide probabilistic admission estimates rather than fragile point predictions.

## Profile: High Achiever
- **Student Rank:** 2000
- **Category:** OPEN
- **Preferences:** IIT - Computer Science

### Top 10 Recommended Options
| Institute | Program | Predicted Cutoff | Admission Prob | 90% CI Bounds | Classification |
|---|---|---|---|---|---|
| Indian Institute of Technology Bhilai | Computer Science and Engineering | 7382 | 100.0% | [7221 - 7542] | Safe |
| Indian Institute of Technology Dharwad | Computer Science and Engineering | 7208 | 100.0% | [7045 - 7354] | Safe |
| Indian Institute of Technology Jammu | Computer Science and Engineering | 6651 | 100.0% | [6469 - 6821] | Safe |
| Indian Institute of Technology Palakkad | Computer Science and Engineering | 6454 | 100.0% | [6289 - 6608] | Safe |
| Indian Institute of Technology Goa | Computer Science and Engineering | 6242 | 100.0% | [6066 - 6413] | Safe |
| Indian Institute of Technology Tirupati | Computer Science and Engineering | 5034 | 100.0% | [4872 - 5217] | Safe |
| Indian Institute of Technology Patna | B.Tech (Computer Science and Engineering) - MBA in Digital Business Management (IIM Bodh Gaya) | 4297 | 100.0% | [4133 - 4467] | Safe |
| Indian Institute of Technology Bhubaneswar | Computer Science and Engineering | 4162 | 100.0% | [4006 - 4324] | Safe |
| Indian Institute of Technology (ISM) Dhanbad | Computer Science and Engineering | 3585 | 100.0% | [3418 - 3759] | Safe |
| Indian Institute of Technology Patna | Computer Science and Engineering | 3377 | 100.0% | [3219 - 3536] | Safe |



## Profile: Mid-Tier Aspirant
- **Student Rank:** 15000
- **Category:** OBC-NCL
- **Preferences:** NIT - Electronics

### Top 10 Recommended Options
| Institute | Program | Predicted Cutoff | Admission Prob | 90% CI Bounds | Classification |
|---|---|---|---|---|---|
| National Institute of Technology Sikkim | Electrical and Electronics Engineering | 476705 | 100.0% | [476542 - 476860] | Safe |
| National Institute of Technology, Manipur | Electronics and Communication Engineering | 196250 | 100.0% | [196087 - 196407] | Safe |
| National Institute of Technology Sikkim | Electronics and Communication Engineering | 172825 | 100.0% | [172658 - 172993] | Safe |
| National Institute of Technology Goa | Electrical and Electronics Engineering | 149436 | 100.0% | [149274 - 149603] | Safe |
| National Institute of Technology, Srinagar | Electronics and Communication Engineering | 145774 | 100.0% | [145615 - 145937] | Safe |
| National Institute of Technology Goa | Electronics and Communication Engineering | 94812 | 100.0% | [94644 - 94966] | Safe |
| National Institute of Technology Hamirpur | Electronics and Communication Engineering | 50316 | 100.0% | [50144 - 50483] | Safe |
| National Institute of Technology Puducherry | Electrical and Electronics Engineering | 48451 | 100.0% | [48287 - 48611] | Safe |
| National Institute of Technology Arunachal Pradesh | Electronics and Communication Engineering | 40278 | 100.0% | [40092 - 40445] | Safe |
| National Institute of Technology Puducherry | Electronics and Communication Engineering | 36972 | 100.0% | [36809 - 37132] | Safe |



## Profile: Targeted State Student
- **Student Rank:** 40000
- **Category:** OPEN
- **Preferences:** NIT - Civil

### Top 10 Recommended Options
| Institute | Program | Predicted Cutoff | Admission Prob | 90% CI Bounds | Classification |
|---|---|---|---|---|---|
| National Institute of Technology, Mizoram | Civil Engineering | 626229 | 100.0% | [626060 - 626400] | Safe |
| National Institute of Technology Sikkim | Civil Engineering | 366110 | 100.0% | [365947 - 366269] | Safe |
| National Institute of Technology, Manipur | Civil Engineering | 287838 | 100.0% | [287658 - 288010] | Safe |
| National Institute of Technology Meghalaya | Civil Engineering | 259471 | 100.0% | [259292 - 259639] | Safe |
| National Institute of Technology Agartala | Civil Engineering | 237333 | 100.0% | [237175 - 237500] | Safe |
| National Institute of Technology Goa | Civil Engineering | 178345 | 100.0% | [178171 - 178511] | Safe |
| National Institute of Technology Goa | Civil Engineering | 168173 | 100.0% | [168014 - 168335] | Safe |
| National Institute of Technology Puducherry | Civil Engineering | 142535 | 100.0% | [142376 - 142692] | Safe |
| National Institute of Technology, Srinagar | Civil Engineering | 135449 | 100.0% | [135289 - 135611] | Safe |
| National Institute of Technology Hamirpur | Civil Engineering | 111968 | 100.0% | [111803 - 112126] | Safe |


