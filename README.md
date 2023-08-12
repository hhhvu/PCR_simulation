# PCR_simulation

## Data
**Data keys:** curve_data, sample_info, igi_gene_call
![](https://github.com/hhhvu/PCR_simulation/blob/main/data/data_schemas.png)
**curve_call**: contains Rn, dRn, and fluorescence signal readings at each cycle for each curve.

* well_position (str): well position with the letters indicating the rows and the numbers indicating the columns of the well e.g. A1, B13, ... . There are 16 rows and 24 columns that make up 384 wells in total.
* target (str): target gene names. For Thermo files, target genes are N gene, S gene, ORF1ab, and MS2. For Luner files, target genes are N gene, E gene and RnaseP.
* dye (str): dye name. Values: FAM, VIC, JUN, ABY, ATTO 647. 
* amp_score (float): amplification score.
* cq (float): cycle number where the curve passed the threshold. If "Undetermined", the curve does not pass the threshold.
* threshold (float): dRn threshold.
* baseline_start (int): cycle number where the baseline calculation starts (usually 3)
* baseline_end (int): cycle number where the baseline calculation ends
* cycle_no (int): amplification cycle number. For Thermo files, the maximum number cycle is 40. For Luner files, the maximum number cycle is 45.
* call (str): IGI label for the gene. Values can be "Absence" or "Presence".
* rn (float): Rn is the fluorescence of the reporter dye divided by the fluorescence of a passive reference dye; i.e.,Rn is the reporter signal normalized to the fluorescence signal of Applied Biosystems™ ROX™ Dye.
* drn (float): maximum delta Rn value of the curve
* pcr_plate (str): qPCR Plate 1 ID. Unique IGI idenitfier for PCR run.
* curve_idx (int): unique identifier created based on pcr_plate, well_position, target groups.

**sample_info**: contains sample results.

* sample_id (str): IGI sample unique identifier.
* sample_barcode (str): IGI sample unique identifier.
* pcr_plate (str): qPCR Plate 1 ID. Unique IGI idenitfier for PCR run.
* well_position (str): well position with the letters indicating the rows and the numbers indicating the columns of the well e.g. A1, B13, ... . There are 16 rows and 24 columns that make up 384 wells in total.
* sample_type (str): Values: clinical samples, buffer negative control, negative control (qPCR), positive control (qPCR), human normal negative control.
* final_patient_result (str): IGI patient sample label. Values: positive, negative, resample.
* current_patient_result (str): IGI current run patient sample result. Values: positive, negative, invalid, inconclusive.
* created_date (datetime): Run date.
* record_type (str): Values: submitted sample, pooled sample, re-test sample, control sample.
* retest_sample_id_1 (str): sample id of the first retest sample if there is.
* retest_sample_id_2 (str): sample id of the second retest sample if there is.

**igi_gene_call**: contain IGI curve information.

* sample_id (str): IGI sample unique identifier.
* pcr_plate (str): qPCR Plate 1 ID. Unique IGI idenitfier for PCR run.
* target (str): target gene names. For Thermo files, target genes are N gene, S gene, ORF1ab, and MS2. For Luner files, target genes are N gene, E gene and RnaseP.
* igi_call (str): Values: positive, negative.
* thres_ct (float): threshold on cycle where a curve should pass dRn threshold to be considered as positive.
  
