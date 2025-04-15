## Dataset
DOTA-H can be downloaded from [https://captain-whu.github.io/DOTA/dataset.html](https://captain-whu.github.io/DOTA/dataset.html)
Download Dota V1.5, use both Training set and Validation set.
You will need both the images and labels.
The Data must be loaded as below. Otherwise, edit the folder relative position in main()
```
raw_data/
├── dota_orig/
│   ├── images/
│   └── labelTxt/
dotah_generator/
├── <generator file>
```