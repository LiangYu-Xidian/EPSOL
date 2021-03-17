# EPSOL: sequence-based protein solubility prediction using multidimensional embedding

## Motivation
The heterologous expression of recombinant protein requires host cells, such as Escherichia coli, and the solubility of protein greatly affects the protein yield. A novel and highly accurate solubility predictor that concurrently improves the production yield and minimizes production cost, and that forecasts protein solubility in an E. coli expression system before the actual experimental work is highly sought.

## Installation

### Requirements

* Install Anaconda (https://www.anaconda.com/download)
  * Create EPSOL environment (Run `conda env create -f tf.yml` )
* SCRATCH-1D release 1.2 (http://download.igb.uci.edu/SCRATCH-1D_1.2.tar.gz)  


* R  requirements (https://www.r-project.org)
	* R libraries
		* bio3d
		* stringr
		* Interpol
		* zoo

You can also create R environment by conda (Run `conda env create -f R.yml` )

Use `conda activate tf` or `conda activate R`  to activate the environment. 



## Run  EPSOL on new test file

You need to perform three steps to predict  new test file (e.g. new_test.fasta).

1. Run SCRATCH with the new test file.
   * Execute in the command line:
	Run `your_SCRATCH_installation_path/bin/run_SCRATCH-1D_predictors.sh 		new_test.fasta new_test 20 `
	`20` is the number of processors, `new_test` is the output files' prefix.
	* It will return four files in current folder: 
	  1. new_test.ss 
	  2. new_test.ss8 
	  3. new_test.acc 
	  4. new_test.acc20
2. Calculate features for test sequences.

   * Execute in the command line: 
	Run `R --vanilla < PaRSnIP.R new_test.fasta new_test.ss new_test.ss8 new_test.acc20 new_test`
   * Following this step, one file is created in current folder:
     1. new_test_src_bio: contains biological features corresponding to the raw protein sequences
3. Run EPSOL prediction file.
    * Execute in the command line: 
      Run `python new_test.fasta new_test.ss new_test.ss8 new_test.acc new_test.acc20 new_test`
    * The final prediction result will be saved in  `./result/predict_file/`, and the filename is `new_test_prediction.txt`



## Contact
Liang Yu:  lyu@xidian.edu.cn
