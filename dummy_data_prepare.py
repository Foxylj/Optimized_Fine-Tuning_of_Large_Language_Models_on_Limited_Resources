import pdb
import utils

PATH_TO_DATA="alpaca_data.json"
TARGET="alpaca_data_dummy_2.json"

data=utils.jload(PATH_TO_DATA)
new_data=data[:200]
utils.jdump(new_data,TARGET)
data=utils.jload(TARGET)



