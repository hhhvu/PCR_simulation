**Data keys:** da_call, amplification, multicomponent, raw_data, da_result

**da_call**:

* well_position (str): well position with the letters indicating the rows and the numbers indicating the columns of the well e.g. A1, B13, ... . There are 16 rows and 24 columns that make up 384 wells in total.
* target (str): target gene names. For Thermo files, target genes are N gene, S gene, ORF1ab, and MS2. For Luner files, target genes are N gene, E gene and RnaseP.
* call (str): IGI label for the gene. Values can be "Absence" or "Presence".
* cq (float): cycle number where the curve passed the threshold. If "Undetermined", the curve does not pass the threshold.
* drn (float): maximum delta Rn value of the curve
* file (str): last 4 characters of qPCR plate 1 barcode

**amplification**:

* well_position (str): well position with the letters indicating the rows and the numbers indicating the columns of the well e.g. A1, B13, ... . There are 16 rows and 24 columns that make up 384 wells in total.
* cycle_no (int): amplification cycle number. For Thermo files, the maximum number cycle is 40. For Luner files, the maximum number cycle is 45.
* target (str): target gene names. For Thermo files, target genes are N gene, S gene, ORF1ab, and MS2. For Luner files, target genes are N gene, E gene and RnaseP.
* rn (float): Rn is the fluorescence of the reporter dye divided by the fluorescence of a passive reference dye; i.e.,Rn is the reporter signal normalized to the fluorescence signal of Applied Biosystems™ ROX™ Dye.
* drn (float): ΔRn is Rn minus the baseline.
* file (str): last 4 characters of qPCR plate 1 barcode

**multicomponent**:

* well_position (str): well position with the letters indicating the rows and the numbers indicating the columns of the well e.g. A1, B13, ... . There are 16 rows and 24 columns that make up 384 wells in total.
* cycle_no (int): amplification cycle number. For Thermo files, the maximum number cycle is 40. For Luner files, the maximum number cycle is 45.
* dye (str): name of dyes used for fluorescence signals; also called "target reporter". Names can be FAM, VIC, ATTO 647 for Luner and FAM, VIC, JUN, ABY for Thermo.
* Fn (float): measured fluorescent signal
* file (str): last 4 characters of qPCR plate 1 barcode

  
**raw_data**:

Columns:
* well_position (str): well position with the letters indicating the rows and the numbers indicating the columns of the well e.g. A1, B13, ... . There are 16 rows and 24 columns that make up 384 wells in total.
* cycle_no (int): amplification cycle number. For Thermo files, the maximum number cycle is 40. For Luner files, the maximum number cycle is 45.
* filter (str): name of optical filter used to read fluorescence signals. Values can be x1-m1, x2-m2, x3-m3, x4-m4 and x5-m5.
* values (float): reading from the optical filters. We want to confirm that the readings from optical filters display increasing characteristic over the cycles.
* file (str): last 4 characters of qPCR plate 1 barcode
  
**da_result**:

Columns:
* well_position (str): well position with the letters indicating the rows and the numbers indicating the columns of the well e.g. A1, B13, ... . There are 16 rows and 24 columns that make up 384 wells in total.
* target (str): target gene names. For Thermo files, target genes are N gene, S gene, ORF1ab, and MS2. For Luner files, target genes are N gene, E gene and RnaseP.
* amp_status (str): indicator if delta Rn curve passed the threshold is detected by the software. Values: Amp, No Amp, Inconclusive, N/A
* amp_score (float): 
* cq (float):
* cq_confidence (float):
* cq_mean (float):
* cq_sd (float): 
* threshold (float): delta Rn threshold which can be set automatically or manually
* baseline_start (int): cycle number where the baseline calculation starts (usually 3)
* baseline_end (int): cycle number where the baseline calculation ends
* file (str): last 4 characters of qPCR plate 1 barcode

