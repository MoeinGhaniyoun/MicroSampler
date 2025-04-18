import pandas as pd
import sys
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

corrs = []
file=sys.argv[1]
units = ['EUU-ALU', 'LQ-ADDR', 'SQ-ADDR', 'LQ-PC', 'SQ-PC', 'ROB-PC']
#unit = 'SQ-PC'
iterss = 100
category='V2'
type_='warmup'

for unit in units:
	#prepare column
	def make_column_name(base,size):
	    arr_name=[]
	    for i in range(size):
	        arr_name.append(base+'_'+str(i))
	    return arr_name

	LQ_PC_arr=make_column_name('LQ_PC', 8)
	LQ_ADDR_arr=make_column_name('LQ_ADDR',8)
	SQ_PC_arr=make_column_name('SQ_PC',8)
	SQ_ADDR_arr=make_column_name('SQ_ADDR',8)

	ROB_PC_arr=make_column_name('ROB_PC',32)
	LFB_DATA_arr = make_column_name('LFB_DATA',16)
	#EUU_BITVEC_arr = make_column_name('EUU_BITVEC',4)
	EUU_ALU_arr = make_column_name('EUU_ALU',2)
	EUU_ADDRG_arr = make_column_name('EUU_ADDRG',1)
	EUU_DIV_arr = make_column_name('EUU_DIV',1)
	EUU_MUL_arr = make_column_name('EUU_MUL',1)

	super_column = ['LQ-OCPNCY', 'LQ-PC', 'LQ-ADDR', 'SQ-OCPNY', 'SQ-PC', 'SQ-ADDR',
	       'ROB-OCPNY', 'ROB-PC', 'LFB-DATA', 'EUU-ALU', 'EUU-ADDRG','EUU-DIV','EUU-MUL','PREF-ADDR'
	       , 'CLASS', 'ITER', 'KEY', 'DTLB-NMISS', 'DCACHE-NMISS']

	super_column_group = [['LQ-OCPNCY'], LQ_PC_arr, LQ_ADDR_arr, ['SQ-OCPNY'], SQ_PC_arr, SQ_ADDR_arr,
	       ['ROB-OCPNY'], ROB_PC_arr, LFB_DATA_arr, EUU_ALU_arr, EUU_ADDRG_arr, EUU_DIV_arr, EUU_MUL_arr,
	                      ['PREF-ADDR'],['CLASS'], ['ITER'],['KEY'], ['DTLB-NMISS'], ['DCACHE-NMISS']]

	# super_column = ['CYCLE', 'LQ-OCPNCY', 'LQ-PC', 'LQ-ADDR', 'SQ-OCPNY', 'SQ-PC', 'SQ-ADDR',
	#        'ROB-OCPNY', 'ROB-PC', 'LFB-DATA', 'EUU-ALU', 'EUU-ADDRG','EUU-DIV','EUU-MUL','PREF-ADDR'
	#        , 'CLASS', 'ITER', 'KEY', 'DTLB-NMISS', 'DCACHE-NMISS']

	# super_column_group = [['CYCLE'], ['LQ-OCPNCY'], LQ_PC_arr, LQ_ADDR_arr, ['SQ-OCPNY'], SQ_PC_arr, SQ_ADDR_arr,
	#        ['ROB-OCPNY'], ROB_PC_arr, LFB_DATA_arr, EUU_ALU_arr, EUU_ADDRG_arr, EUU_DIV_arr, EUU_MUL_arr,
	#                       ['PREF-ADDR'],['CLASS'], ['ITER'],['KEY'], ['DTLB-NMISS'], ['DCACHE-NMISS']]
	sub_column=[]
	for arr in super_column_group:
	    sub_column.extend(arr)

	standard_scaling_feature=['LQ-OCPNCY','SQ-OCPNY','ROB-OCPNY']
	PC_features=['LQ-PC', 'SQ-PC','ROB-PC']
	add_features=['LQ-ADDR','SQ-ADDR','PREF-ADDR']
	euu_features=['EUU-ALU', 'EUU-ADDRG','EUU-DIV','EUU-MUL']

	df = pd.read_csv(file,sep=',',names = sub_column,header=1)#[0:1000]



	df_list=[]
	for i,group in enumerate(super_column_group):
	    temp_df = df[group]
	    temp_df.columns = pd.MultiIndex.from_product([[super_column[i]], temp_df.columns])
	    #display(temp_df.head())
	    df_list.append(temp_df)
	new_df = pd.concat(df_list,axis=1)
	new_df.drop([('LFB-DATA',), ('DTLB-NMISS',), ('DCACHE-NMISS',)],axis=1,inplace=True)


	new_df = new_df[[unit, 'CLASS', 'ITER', 'KEY']]
	subsets=[]
	for items in new_df[unit]:
	    subsets.append((unit, items))
	#new_df = new_df.dropna(subset=subsets, how='all')
	new_df.fillna('0x00',inplace=True)

	addr_dict={}
	def addr_dict_maker(x):
	    addr_dict[x]=None

	new_df[unit].applymap(addr_dict_maker)
	ind=1
	for i,key in enumerate(addr_dict):
	    if key!='0x00':
	        addr_dict[key] = ind
	        ind+=1
	    elif key=='0x00':
	        addr_dict[key] = 0

	#print("Feature Dictionary:")
	#addr_dict



	itr = [i for i in range(iterss)]
	keys = ['0x44','0xaa','rand-0.10_0.90','rand-0.20_0.80','rand-0.30_0.70','rand-0.40_0.60','rand-0.50_0.50',
	        'rand-0.60_0.40','rand-0.70_0.30','rand-0.80_0.20','rand-0.90_0.10']
	#keys = ['rand-0.50_0.50']
	c=0
	i=0
	X_inp = {}
	Y_inp = {}
	for k in keys:
	    X_inp[k] = []
	    Y_inp[k] = []

	tmp=[]
	curr_key = keys[0]
	curr_iter = 0

	for i in range(len(new_df)):
	    if(new_df[('KEY','KEY')].iloc[i]!=curr_key):
	        X_inp[curr_key].append(np.array(tmp))
	        Y_inp[curr_key].append(np.array([new_df[('CLASS','CLASS')].iloc[i-1]]))
	        curr_key = new_df[('KEY','KEY')].iloc[i]
	        curr_iter = new_df[('ITER','ITER')].iloc[i]
	        tmp=[]
	        #break
	    if(new_df[('ITER','ITER')].iloc[i]!=curr_iter):
	        curr_iter = new_df[('ITER','ITER')].iloc[i]
	        X_inp[curr_key].append(np.array(tmp))
	        Y_inp[curr_key].append(np.array([new_df[('CLASS','CLASS')].iloc[i-1]]))
	        tmp=[]
	    if unit == 'EUU-ALU':
	      tmp.append([addr_dict[new_df[('EUU-ALU','EUU_ALU_0')].iloc[i]],
	                 addr_dict[new_df[('EUU-ALU','EUU_ALU_1')].iloc[i]],])
	    elif unit == 'LQ-ADDR':
	      tmp.append([addr_dict[new_df[('LQ-ADDR','LQ_ADDR_0')].iloc[i]],
	                  addr_dict[new_df[('LQ-ADDR','LQ_ADDR_1')].iloc[i]],
	                  addr_dict[new_df[('LQ-ADDR','LQ_ADDR_2')].iloc[i]],
	                  addr_dict[new_df[('LQ-ADDR','LQ_ADDR_3')].iloc[i]],
	                  addr_dict[new_df[('LQ-ADDR','LQ_ADDR_4')].iloc[i]],
	                  addr_dict[new_df[('LQ-ADDR','LQ_ADDR_5')].iloc[i]],
	                  addr_dict[new_df[('LQ-ADDR','LQ_ADDR_6')].iloc[i]],
	                  addr_dict[new_df[('LQ-ADDR','LQ_ADDR_7')].iloc[i]],])
	    elif unit == 'SQ-ADDR':
	      tmp.append([addr_dict[new_df[('SQ-ADDR','SQ_ADDR_0')].iloc[i]],
	                addr_dict[new_df[('SQ-ADDR','SQ_ADDR_1')].iloc[i]],
	                addr_dict[new_df[('SQ-ADDR','SQ_ADDR_2')].iloc[i]],
	                addr_dict[new_df[('SQ-ADDR','SQ_ADDR_3')].iloc[i]],
	                addr_dict[new_df[('SQ-ADDR','SQ_ADDR_4')].iloc[i]],
	                addr_dict[new_df[('SQ-ADDR','SQ_ADDR_5')].iloc[i]],
	                addr_dict[new_df[('SQ-ADDR','SQ_ADDR_6')].iloc[i]],
	                addr_dict[new_df[('SQ-ADDR','SQ_ADDR_7')].iloc[i]],])
	    elif unit == 'LQ-PC':
	      tmp.append([addr_dict[new_df[('LQ-PC','LQ_PC_0')].iloc[i]],
	            addr_dict[new_df[('LQ-PC','LQ_PC_1')].iloc[i]],
	            addr_dict[new_df[('LQ-PC','LQ_PC_2')].iloc[i]],
	            addr_dict[new_df[('LQ-PC','LQ_PC_3')].iloc[i]],
	            addr_dict[new_df[('LQ-PC','LQ_PC_4')].iloc[i]],
	            addr_dict[new_df[('LQ-PC','LQ_PC_5')].iloc[i]],
	            addr_dict[new_df[('LQ-PC','LQ_PC_6')].iloc[i]],
	            addr_dict[new_df[('LQ-PC','LQ_PC_7')].iloc[i]],])
	    elif unit == 'SQ-PC':
	      tmp.append([addr_dict[new_df[('SQ-PC','SQ_PC_0')].iloc[i]],
	            addr_dict[new_df[('SQ-PC','SQ_PC_1')].iloc[i]],
	            addr_dict[new_df[('SQ-PC','SQ_PC_2')].iloc[i]],
	            addr_dict[new_df[('SQ-PC','SQ_PC_3')].iloc[i]],
	            addr_dict[new_df[('SQ-PC','SQ_PC_4')].iloc[i]],
	            addr_dict[new_df[('SQ-PC','SQ_PC_5')].iloc[i]],
	            addr_dict[new_df[('SQ-PC','SQ_PC_6')].iloc[i]],
	            addr_dict[new_df[('SQ-PC','SQ_PC_7')].iloc[i]],])

	    elif unit == 'ROB-PC':
	      tmp.append([addr_dict[new_df[('ROB-PC','ROB_PC_0')].iloc[i]],
	           addr_dict[new_df[('ROB-PC','ROB_PC_1')].iloc[i]],
	           addr_dict[new_df[('ROB-PC','ROB_PC_2')].iloc[i]],
	           addr_dict[new_df[('ROB-PC','ROB_PC_3')].iloc[i]],
	           addr_dict[new_df[('ROB-PC','ROB_PC_4')].iloc[i]],
	           addr_dict[new_df[('ROB-PC','ROB_PC_5')].iloc[i]],
	           addr_dict[new_df[('ROB-PC','ROB_PC_6')].iloc[i]],
	           addr_dict[new_df[('ROB-PC','ROB_PC_7')].iloc[i]],
	           addr_dict[new_df[('ROB-PC','ROB_PC_8')].iloc[i]],
	           addr_dict[new_df[('ROB-PC','ROB_PC_9')].iloc[i]],
	           addr_dict[new_df[('ROB-PC','ROB_PC_10')].iloc[i]],
	           addr_dict[new_df[('ROB-PC','ROB_PC_11')].iloc[i]],
	           addr_dict[new_df[('ROB-PC','ROB_PC_12')].iloc[i]],
	           addr_dict[new_df[('ROB-PC','ROB_PC_13')].iloc[i]],
	           addr_dict[new_df[('ROB-PC','ROB_PC_14')].iloc[i]],
	           addr_dict[new_df[('ROB-PC','ROB_PC_15')].iloc[i]],
	           addr_dict[new_df[('ROB-PC','ROB_PC_16')].iloc[i]],
	           addr_dict[new_df[('ROB-PC','ROB_PC_17')].iloc[i]],
	           addr_dict[new_df[('ROB-PC','ROB_PC_18')].iloc[i]],
	           addr_dict[new_df[('ROB-PC','ROB_PC_19')].iloc[i]],
	           addr_dict[new_df[('ROB-PC','ROB_PC_20')].iloc[i]],
	           addr_dict[new_df[('ROB-PC','ROB_PC_21')].iloc[i]],
	           addr_dict[new_df[('ROB-PC','ROB_PC_22')].iloc[i]],
	           addr_dict[new_df[('ROB-PC','ROB_PC_23')].iloc[i]],
	           addr_dict[new_df[('ROB-PC','ROB_PC_24')].iloc[i]],
	           addr_dict[new_df[('ROB-PC','ROB_PC_25')].iloc[i]],
	           addr_dict[new_df[('ROB-PC','ROB_PC_26')].iloc[i]],
	           addr_dict[new_df[('ROB-PC','ROB_PC_27')].iloc[i]],
	           addr_dict[new_df[('ROB-PC','ROB_PC_28')].iloc[i]],
	           addr_dict[new_df[('ROB-PC','ROB_PC_29')].iloc[i]],
	           addr_dict[new_df[('ROB-PC','ROB_PC_30')].iloc[i]],
	           addr_dict[new_df[('ROB-PC','ROB_PC_31')].iloc[i]],])


	print (curr_key)
	X_inp[curr_key].append(np.array(tmp))
	Y_inp[curr_key].append(np.array([new_df[('CLASS','CLASS')].iloc[i-1]]))
	print('Done')



	m = -1
	for keys in X_inp:
	    for i in range(iterss):
	        #print(keys, X_inp[keys][i].shape, Y_inp[keys][i])
	        if(m<X_inp[keys][i].shape[0]):
	            m=X_inp[keys][i].shape[0]

	print('max cycle: ', m)
	#print('sample: ', X_inp['0x44'][0].shape[0])

	m1 = X_inp['rand-0.50_0.50'][0].shape[0]

	for keys in X_inp:
	    for i in range(iterss):
	        #print(keys, X_inp[keys][i].shape, Ys_inp[keys][i])
	        if(m1>X_inp[keys][i].shape[0]):
	            m1=X_inp[keys][i].shape[0]

	print('min cycle: ', m1)
	print('sample: ', X_inp['rand-0.50_0.50'][0].shape[0])


	#max
	for keys in X_inp:
	    for i in range(iterss):
	        for j in range(m - X_inp[keys][i].shape[0]):
	            tmp=[]
	            for k in range(X_inp[keys][i].shape[1]):
	                tmp.append(0)
	            X_inp[keys][i] = np.append(X_inp[keys][i], [tmp], axis=0)


	groups = new_df.groupby([('ITER','ITER'),('KEY','KEY')])
	count=[]
	for i,table in groups:
	    #print(i, table.shape[0])
	    count.append(table.shape[0])

	import numpy as np
	from sklearn.metrics import r2_score
	from scipy.stats import chi2_contingency
	train_series = ['0xaa','0x44','rand-0.20_0.80','rand-0.80_0.20','rand-0.40_0.60','rand-0.60_0.40','rand-0.10_0.90','rand-0.90_0.10',
	                'rand-0.30_0.70','rand-0.70_0.30','rand-0.50_0.50']
	#train_series = ['rand-0.50_0.50']
	test_series = ['rand-0.30_0.70','rand-0.70_0.30','rand-0.50_0.50']
	X_train=[]
	X_test=[]
	y_train=[]
	y_test=[]

	for k in train_series:
	    for i in range(iterss):
	        X_train.append(X_inp[k][i])
	        y_train.append(Y_inp[k][i])

	#for k in test_series:
	#    for i in range(2000):
	#        X_test.append(X_inp[k][i])
	#        y_test.append(Y_inp[k][i])

	X_train = np.array(X_train)
	y_train = np.array(y_train)
	#X_test = np.array(X_test)
	#y_test = np.array(y_test)
	print(y_train.shape)
	y_train = y_train.reshape(iterss, -1).ravel()

	print(X_train.shape)



	data = X_train
	labels = y_train

	# Separate the data based on the labels
	data_label_1 = data[y_train == 1]
	data_label_0 = data[y_train == 0]

	# Flatten the arrays corresponding to each label
	flattened_data_label_1 = data_label_1.flatten()
	flattened_data_label_0 = data_label_0.flatten()

	# Find unique values in each label
	unique_values_label_1 = np.unique(flattened_data_label_1)
	unique_values_label_0 = np.unique(flattened_data_label_0)

	# Find values unique to label 1
	values_unique_to_label_1 = np.setdiff1d(unique_values_label_1, unique_values_label_0)

	# Find values unique to label 0
	values_unique_to_label_0 = np.setdiff1d(unique_values_label_0, unique_values_label_1)

	print("Store queue address IDs unique to Key bit 1:")
	print(values_unique_to_label_1)

	print("Store queue address IDs unique to Key bit 0:")
	print(values_unique_to_label_0)

	print("Feature Dictionary:")
	addr_dict

	hashable_arrays = [tuple(map(tuple, arr)) for arr in X_train]

	# Create a dictionary to store the unique scalar values for each array
	array_to_scalar = {}

	# Generate unique scalar values for each array
	for arr in hashable_arrays:
	    if arr not in array_to_scalar:
	        array_to_scalar[arr] = hash(arr)
	X_scalar = np.array([array_to_scalar[tuple(map(tuple, arr))] for arr in X_train])

	print(X_scalar[1])
	#print(X_train.shape)
	print(X_scalar.shape)
	print(y_train.shape)

	import numpy as np
	from sklearn.metrics import r2_score
	from scipy.stats import chi2_contingency
	import matplotlib.pyplot as plt

	# Get unique values and their indices
	unique_values, unique_indices = np.unique(X_scalar, return_inverse=True)

	# Create a dictionary to store the mapping of unique values to IDs
	value_to_id = {value: idx + 1 for idx, value in enumerate(unique_values)}

	# Map the values to IDs
	X_id = np.array([value_to_id[value] for value in X_scalar])
	print(X_id.shape)
	print(y_train)
	num_unique_values = len(unique_values)
	print("Number of unique values:", num_unique_values)



	from tabulate import tabulate
	import scipy.stats as stats
	# Create dictionaries to store counts for labels 0 and 1
	count_0 = {}
	count_1 = {}

	# Iterate through the vector and labels to count occurrences
	for value, label in zip(X_id, y_train):
	    if label == 0:
	        if value in count_0:
	            count_0[value] += 1
	        else:
	            count_0[value] = 1
	    elif label == 1:
	        if value in count_1:
	            count_1[value] += 1
	        else:
	            count_1[value] = 1

	# Get the 10 values with the most count for label 0
	top_10_values_0 = sorted(count_0, key=count_0.get, reverse=True)[:10]

	# Get the 10 values with the most count for label 1
	top_10_values_1 = sorted(count_1, key=count_1.get, reverse=True)[:10]


	contingency_table = []
	row0 = []
	row1 = []

	# Prepare the data for the table
	table_data = []
	for value in set(X_id):  # Use set to ensure unique values
	    count_label_0 = count_0.get(value, 0)
	    count_label_1 = count_1.get(value, 0)
	    # Create a contingency table with counts of IDs for class 0 and class 1
	    row0.append(count_label_0)
	    row1.append(count_label_1)
	    table_data.append([value, count_label_0, count_label_1])

	# Define the headers
	headers = ["Microarchitectural State", "Count (Key bit 0)", "Count (Key bit 1)"]

	# Print the table
	print(tabulate(table_data, headers=headers, tablefmt="grid", stralign="left"))




	#print(row0)
	contingency_table.append(row0)
	contingency_table.append(row1)
	#print(contingency_table)


	# Create an array with two rows and 1100 columns
	#contingency_table = np.zeros((2, 1100))

	# Set the values in an alternating pattern
	#contingency_table[0, ::2] = 1  # Set 1 in the first row, starting from the first column
	#contingency_table[1, 1::2] = 1
	#print(contingency_table)


	'''
	# Print all the shared hashes
	print("Here are all the shared hashes and their values")
	for value in set(X_id):  # Use set to ensure unique values
	    count_label_0 = count_0.get(value, 0)
	    count_label_1 = count_1.get(value, 0)
	    if (count_label_1 > 0 and count_label_0 > 0):
	      print(f"{value} | {count_label_0} | {count_label_1}") '''

	contingency_table = np.array(contingency_table)
	 # Perform the Chi-squared test for independence
	chi2, p, dof, expected = stats.chi2_contingency(contingency_table)

	# Set your significance level (alpha)
	alpha = 0.05
	N = np.sum(contingency_table)

	# Calculate Cramér's V
	k = contingency_table.shape[0]
	r = contingency_table.shape[1]
	cramer_v = np.sqrt(chi2 / (N * (k - 1)))

	# Calculate Cramér's V with bias correction

	k = k - ((k - 1)**2) / (N - 1)
	r = r - ((r - 1)**2) / (N - 1)
	chi2_bar = max(0, cramer_v - (((k - 1) * (r - 1)) / (N - 1)))
	cramer_v_unbiased = np.sqrt(chi2_bar / min(k-1, r-1))




	# Apply Yates's correction
	# Initialize an array for Yates's corrections for each cell
	corrections = np.zeros_like(contingency_table, dtype=float)

	# Apply Yates's correction to each cell
	for i in range(contingency_table.shape[0]):
	    for j in range(contingency_table.shape[1]):
	        corrections[i, j] = ((abs(contingency_table[i, j] - expected[i, j]) - 0.5)**2) / expected[i, j]

	# Apply the corrections to the chi-squared statistic
	chi2_corrected = corrections.sum()

	# Calculate the p-value with Yates's correction
	p_corrected = 1 - stats.chi2.cdf(chi2_corrected, dof)

	#print("Chi-squared (corrected):", chi2_corrected)
	#print("P-value (corrected):", p_corrected)






	print("Cramér's V:", cramer_v)
	#print("Cramér's V Unbiased:", cramer_v_unbiased)
	#print("chai2: ", chi2)
	#print ("p-value: ", p)
	#print ("degree of freedom: ", dof)
	# Compare the p-value to the significance level
	if cramer_v > 0.25:
	    print("There is a statistically significant association between IDs and class labels.")
	    corrs.append(cramer_v)
	else:
	    print("There is no statistically significant association between IDs and class labels.")
	print("#################################################################")
	print("#################################################################")

print("#################################################################")
print("#################################################################")
print("#################################################################")
print("#################################################################")
print("#################################################################")
print("#################################################################")
print("#################################################################")
print("#################################################################")
i = 0
print("Measured correlation for:")
for unit in units:
	print(f'{unit}: {corrs[i]}')
	i += 1