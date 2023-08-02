**Data keys:** igi_result, amplification, multicomponent, raw_data, da_result

**igi_result**:
Columns:
* well_position (str): well position with the letters indicating the rows and the numbers indicating the columns of the well e.g. A1, B13, ... . There are 16 rows and 24 columns that make up 384 wells in total.
* omit (bool): indicator whether the well should be omitted or not
* target (str): target gene names. For Thermo files, target genes are N gene, S gene, ORF1ab, and MS2. For Luner files, target genes are N gene, E gene and RnaseP.
* call (str): IGI label for the gene. Values can be "Absence" or "Presence".
* cq (float): cycle number where the curve passed the threshold. If "Undetermined", the curve does not pass the threshold.
* drn (float): maximum delta Rn value of the curve
* file (str): unique ID of downloaded eds file

**amplification**:
Columns:
* well_position (str): well position with the letters indicating the rows and the numbers indicating the columns of the well e.g. A1, B13, ... . There are 16 rows and 24 columns that make up 384 wells in total.
* cycle_no (int): amplification cycle number. For Thermo files, the maximum number cycle is 40. For Luner files, the maximum number cycle is 45.
* target (str): target gene names. For Thermo files, target genes are N gene, S gene, ORF1ab, and MS2. For Luner files, target genes are N gene, E gene and RnaseP.
* rn (float): Rn is the fluorescence of the reporter dye divided by the fluorescence of a passive reference dye; i.e.,Rn is the reporter signal normalized to the fluorescence signal of Applied Biosystems™ ROX™ Dye.
* drn (float): ΔRn is Rn minus the baseline.
* file (str): unique ID of downloaded eds file

**multicomponent**:
Columns:
*

**raw_data**:
Columns:
*

**da_result**:
Columns:
*

