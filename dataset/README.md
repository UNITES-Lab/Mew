## **Dataset**

1. Please download the following raw data here: [Enable Medicine](https://app.enablemedicine.com/portal/atlas-library/studies/92394a9f-6b48-4897-87de-999614952d94?sid=1168)
- UPMC-HNC: `upmc_raw_data.zip` and `upmc_labels.csv`
- Stanford-CRC: `charville_raw_data.zip` and `charville_labels.csv`
- DFCI-HNC: `dfci_raw_data.zip` and `dfci_labels.csv`

2. After the download is complete, please locate the above files as follows:
```
dataset/
├── charville_data
    └── charville_raw_data.zip
    └── charville_labels.csv
├── upmc_data
    └── upmc_raw_data.zip
    └── upmc_labels.csv
├── dfci_data
    └── dfci_labels.csv
├── general_data
    └── upmc_raw_data.zip
    └── dfci_raw_data.zip
```

3. By running each dataset with a certain task, the preprocessing (which would take a few hours to generate graphs) will automatically happen. Finally, the preprocessed structure for charville data will be as follows:
```
dataset/
├── charville_data
    └── charville_raw_data.zip
    └── charville_labels.csv
    └── dataset_mew
        └── fig
        └── graph
        └── model
        └── tg_graph
    └── raw_data
```
