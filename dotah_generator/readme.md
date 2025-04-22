## Dataset
DOTA-H can be downloaded from [https://captain-whu.github.io/DOTA/dataset.html](https://captain-whu.github.io/DOTA/dataset.html)
Download Dota V1.5, use both Training set and Validation set.
You will need both the images and labels (non-hbb version).
The Data must be loaded as below. Otherwise, edit the folder relative position in main()

You will need joblib for Parallelize all-core to process fast.
This use CPU only.
```
raw_data/
├── dota_orig/
│   ├── train
│   │  ├── images/
│   │  └── labelTxt/
│   ├── val
│   │  ├── images/
│   │  └── labelTxt/
│   └── test
│      └── images/
dotah_generator/
└── <generator file>
```