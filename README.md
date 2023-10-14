# Usage


## Install
```sh
pip install jijmodeling jijzept numpy pandas pydantic -U
```

## obtain data
```sh
python code_with_jijmodeling_graviton.py
```
This script outputs `result.csv`
Note that the following file `config.toml` should be placed in the same directory as `code_with_jijmodeling_graviton.py` for JijZept authentication.
```toml
[default]
url = "https://api.jijzept.com"
token = "<Your JijZept API Key>"
```



## generate plot
```sh
python visualize.py
```
This script outputs png file for grpah.

