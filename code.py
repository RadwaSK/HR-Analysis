import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing 
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier as DT
import os

##########################################################################
######################### DATA PREPROCESSING #############################
##########################################################################
df = pd.read_csv('train.csv')
df = df.replace(r'^\s*$', np.NaN, regex=True)
df.education.fillna('No Education', inplace=True)
df.previous_year_rating.fillna(0, inplace=True)
reg_uniques = np.arange(35)[1:]
for r in reg_uniques:
	name = 'region_' + str(r)
	df.replace(name, r, inplace=True)

all_size = df.shape[0]
pred = df.is_promoted
train_size = int(0.8 * all_size)
test_size = all_size - train_size

df_manipulated = df.copy()

# Converting string data to integers
dep_uniques = df_manipulated.department.unique()
for i, u in enumerate(dep_uniques):
	df_manipulated['department'].replace(u, i, inplace=True)

edu_uniques = df.education.unique()
edu_uniques[2], edu_uniques[3] = edu_uniques[3], edu_uniques[2]
for i, u in enumerate(edu_uniques):
	df_manipulated['education'].replace(u, i, inplace=True)

rec_uniques = df_manipulated.recruitment_channel.unique()
for i, u in enumerate(rec_uniques):
	df_manipulated['recruitment_channel'].replace(u, i, inplace=True)

df_manipulated['gender'].replace('m', 0, inplace=True)
df_manipulated['gender'].replace('f', 1, inplace=True)


# Normalizing Data
data = df_manipulated.values
column_names = df_manipulated.columns
min_max_scaler = preprocessing.MinMaxScaler()
data_scaled = min_max_scaler.fit_transform(data)
df_manipulated = pd.DataFrame(data_scaled, columns=column_names)

# Splitting Data
df_test = df_manipulated.iloc[train_size:,:]
df_manipulated = df_manipulated.iloc[:train_size,:]
train_pred = np.array(pred[:train_size])
test_real_pred = np.array(pred[train_size:])

promoted_indices = df_manipulated.index[df_manipulated['is_promoted'] == 1]
df_manipulated.drop(['employee_id', 'is_promoted'], axis=1, inplace=True)
df_test.drop(['employee_id', 'is_promoted'], axis=1, inplace=True)

print("\nTraining Dataset Size is", df_manipulated.shape)
print("Test Dataset size is", df_test.shape)
print('\nColumns in dataset are:', df.columns.values)
pd.options.display.max_columns = None
print('\nSample of the data:\n\n', df.sample(3), '\n')
pd.options.display.max_columns = 4

##########################################################################
#################### DATA VISUALIZATION & INSIGHTS #######################
##########################################################################
def run_data_visualizations():
	if not os.path.exists('visualizations'):
		os.mkdir('visualizations')

	females_count, males_count = len(df.gender[df.gender == 'f']), len(df.gender[df.gender == 'm'])
	females_ratio = int(np.around(females_count / train_size * 100))
	males_ratio = int(np.around(males_count / train_size * 100))
	print("Percentage of females in the dataset = ", females_ratio, '%', sep='')
	print("Percentage of males in the dataset = ", males_ratio, '%', sep='')

	plt.pie([females_ratio, males_ratio], labels=['Females ' + str(females_ratio) + '%', 'Males ' + str(males_ratio) + '%'])
	plt.title('Percentage of males and females among all employees')
	plt.savefig('visualizations/females_males_pie_chart')
	plt.clf()
	# ----------------------------------------------------------------------------------------------
	# ----------------------------------------------------------------------------------------------
	all_promoted = df[df['is_promoted'] == 1]
	number_of_promotions = all_promoted.shape[0]
	print('\nPercentage of promotions in dataset = ', int(np.around(number_of_promotions / train_size * 100)), '%', sep='')
	males_promoted = all_promoted[all_promoted['gender'] == 'm'].shape[0]
	females_promoted = all_promoted[all_promoted['gender'] == 'f'].shape[0]
	females_promoted_percentage = int(np.around(females_promoted / number_of_promotions * 100))
	males_promoted_percentage = int(np.around(males_promoted / number_of_promotions * 100))
	print("\nPercentage of promoted females among all promoted = ", females_promoted_percentage, '%', sep='')
	print("Percentage of promoted males among all promotes = ", males_promoted_percentage, '%', sep='')

	plt.pie([females_promoted_percentage, males_promoted_percentage], labels=['Females ' + str(females_promoted_percentage) + '%', 'Males ' + str(males_promoted_percentage) + '%'])
	plt.title('Among all promoted')
	plt.savefig('visualizations/females_males_promoted_pie_chart')
	plt.clf()
	# ----------------------------------------------------------------------------------------------
	# ----------------------------------------------------------------------------------------------
	females_promoted_ratio = int(np.around(females_promoted / females_count * 100))
	males_promoted_ratio = int(np.around(males_promoted / males_count * 100))
	ratio = int(np.around(females_promoted_ratio / (females_promoted_ratio+males_promoted_ratio) * 100))
	print("\nPercentage of promoted females to all females = ", females_promoted_ratio, '%', sep='')
	print("Percentage of promoted males to all males = ", males_promoted_ratio, '%', sep='')

	plt.pie([females_promoted_ratio, males_promoted_ratio], labels=['Females ' + str(ratio) + '%', 'Males ' + str(100 - ratio) + '%'])
	plt.title('Ratio between promoted males and females to all males and females')
	plt.savefig('visualizations/females_males_promoted_ratio_pie_chart')
	plt.clf()
	# ----------------------------------------------------------------------------------------------
	# ----------------------------------------------------------------------------------------------
	award_won_prom_corr = df_manipulated['awards_won?'].corr(df.is_promoted)
	print("\nCorrelation between Whether an Award is won and Promotion is", award_won_prom_corr)

	award_promoted = all_promoted[all_promoted['awards_won?'] == 1].shape[0]
	no_award_but_promoted = all_promoted[all_promoted['awards_won?'] == 0].shape[0]
	award_percentage = int(np.around((award_promoted+no_award_but_promoted) / train_size * 100))
	award_promoted_percentage = int(np.around(award_promoted / number_of_promotions * 100))
	no_award_but_promoted_percentage = int(np.around(no_award_but_promoted / number_of_promotions * 100))
	print('\nPercentage of employees who won awards = ', award_percentage, '%', sep='')
	print("Percentage of promoted who won awards among all promoted = ", award_promoted_percentage, '%', sep='')
	print("Percentage of promoted who didn't win awards among all promotes = ", no_award_but_promoted_percentage, '%', sep='')

	plt.pie([award_promoted_percentage, no_award_but_promoted_percentage], labels=['Awards Won ' + str(award_promoted_percentage) + '%', 'Awards Not Won ' + str(no_award_but_promoted_percentage) + '%'])
	plt.title("Among all who got promoted")
	plt.savefig('visualizations/award_won_prom_pie_chart')
	plt.clf()

	all_won = df[df['awards_won?'] == 1].shape[0]
	won_not_promoted = all_won - award_promoted
	award_promoted_percentage = int(np.around(award_promoted / all_won * 100))
	award_but_not_promoted_percentage = int(np.around(won_not_promoted / all_won * 100))
	print("\nPercentage of who won awards and promoted among all won = ", award_promoted_percentage, '%', sep='')
	print("Percentage of who won awards and not promoted among all won = ", award_but_not_promoted_percentage, '%', sep='')

	plt.pie([award_promoted_percentage, award_but_not_promoted_percentage], labels=['Promoted ' + str(award_promoted_percentage) + '%', 'Not Promoted ' + str(award_but_not_promoted_percentage) + '%'])
	plt.title("Among all who won awards")
	plt.savefig('visualizations/won_prom_pie_chart')
	plt.clf()
	# ----------------------------------------------------------------------------------------------
	# ----------------------------------------------------------------------------------------------
	age_prom_corr = df.age.corr(df.is_promoted)
	print("\nCorrelation between age and promotion is", age_prom_corr)

	ages = df[['age']].values.flatten()
	max_age = ages.max()
	min_age = ages.min()

	print('\nMaximum age is', max_age)
	print('\nMinimum age is', min_age)

	prom = df[['is_promoted']].values.flatten()

	ages_prom_hist = np.zeros((2, 8))
	for i in range(train_size):
		index = int((ages[i] - 20) / 5)
		index = min(index, 7)
		ages_prom_hist[prom[i], index] += 1


	labels = ['20-24', '25-29', '30-34', '35-39', '40-24', '45-29', '50-34', '55-60']

	x_axis = np.arange(8)
	plt.bar(x_axis - 0.2, ages_prom_hist[0], 0.4, label='Not Promoted')
	plt.bar(x_axis + 0.2, ages_prom_hist[1], 0.4, label='Promoted')
	plt.xticks(x_axis, labels)
	plt.xlabel('Ages')
	plt.ylabel('Histogram')
	plt.title('Number of promoted in each age range')
	plt.legend()
	plt.savefig('visualizations/age_prom_barplot')
	plt.clf()
	# ----------------------------------------------------------------------------------------------
	# ----------------------------------------------------------------------------------------------
	edu_prom_corr = df_manipulated.education.corr(df.is_promoted)
	print("\nCorrelation between education and promotion is", edu_prom_corr)

	education = df[['education']].values.flatten()

	edus_prom_hist = np.zeros((2, 4))
	for i in range(train_size):
		j = np.where(edu_uniques == education[i])
		edus_prom_hist[prom[i], j] += 1


	x_axis = np.arange(4)
	plt.bar(x_axis - 0.2, edus_prom_hist[0], 0.4, label='Not Promoted')
	plt.bar(x_axis + 0.2, edus_prom_hist[1], 0.4, label='Promoted')
	plt.xticks(x_axis, edu_uniques)
	plt.xlabel('Education')
	plt.ylabel('Histogram')
	plt.title('Number of promoted in each education category')
	plt.legend()
	plt.savefig('visualizations/edu_prom_barplot')
	plt.clf()
	# ----------------------------------------------------------------------------------------------
	# ----------------------------------------------------------------------------------------------
	dep_prom_corr = df_manipulated.department.corr(df.is_promoted)
	print("\nCorrelation between department and promotion is", dep_prom_corr)

	deps = df[['department']].values.flatten()

	deps_prom_hist = np.zeros((2, len(dep_uniques)))
	for i in range(train_size):
		j = np.where(dep_uniques == deps[i])
		deps_prom_hist[prom[i], j] += 1

	x_axis = np.arange(len(dep_uniques))
	plt.bar(x_axis - 0.2, deps_prom_hist[0], 0.4, label='Not Promoted')
	plt.bar(x_axis + 0.2, deps_prom_hist[1], 0.4, label='Promoted')
	plt.xticks(x_axis, dep_uniques)
	plt.xticks(rotation=45)
	plt.xlabel('Departments')
	plt.ylabel('Histogram')
	plt.title('Number of promoted in each department')
	plt.legend()
	plt.tight_layout()
	plt.savefig('visualizations/dep_prom_barplot')
	plt.clf()
	# ----------------------------------------------------------------------------------------------
	# ----------------------------------------------------------------------------------------------
	avg_sc_prom_corr = df_manipulated.avg_training_score.corr(df.is_promoted)
	print("\nCorrelation between Average Training Score and Promotion is", avg_sc_prom_corr)

	avg_scores = df[['avg_training_score']].values.flatten()
	max_score = avg_scores.max()
	min_score = avg_scores.min()

	print('\nMaximum Average Training Score is', max_score)
	print('\nMinimum Average Training Score is', min_score)


	avg_scores_prom_hist = np.zeros((2, 12))
	for i in range(train_size):
		index = int((avg_scores[i] - 39) / 5)
		index = min(index, 11)
		avg_scores_prom_hist[prom[i], index] += 1


	labels = ['39-43', '44-48', '49-53', '54-58', '59-63', '64-68', '69-73', '74-78', '79-83', '84-88', '89-93', '94-99']

	for i in range(12):
		r = int(np.around((avg_scores_prom_hist[1, i] / (avg_scores_prom_hist[0, i] +  avg_scores_prom_hist[1, i])) * 100))
		t = 'Percentage of Promoted with Avg. Training score in [' + labels[i] + '] = ' + str(r) + '%'
		print('\n', t)

	x_axis = np.arange(12)
	plt.bar(x_axis - 0.2, avg_scores_prom_hist[0], 0.4, label='Not Promoted')
	plt.bar(x_axis + 0.2, avg_scores_prom_hist[1], 0.4, label='Promoted')
	plt.xticks(x_axis, labels)
	plt.xticks(rotation=90)
	plt.tick_params(axis='x', which='major', labelsize=7)
	plt.xlabel('Average Training Score')
	plt.ylabel('Histogram')
	plt.title('Number of promoted associated with Average Trainign Score')
	plt.legend()
	plt.tight_layout()
	plt.savefig('visualizations/avg_score_prom_barplot')
	plt.clf()
	# ----------------------------------------------------------------------------------------------
	# ----------------------------------------------------------------------------------------------
	training_num_uniques = sorted(df.no_of_trainings.unique())
	training_num_prom_corr = df_manipulated.department.corr(df.is_promoted)
	print("\nCorrelation between Number of Trainings and Promotion is", training_num_prom_corr)

	training_nums = df[['no_of_trainings']].values.flatten()

	training_nums_prom_hist = np.zeros((2, len(training_num_uniques)))
	for i in range(train_size):
		j = np.where(training_num_uniques == training_nums[i])
		training_nums_prom_hist[prom[i], j] += 1

	x_axis = np.arange(len(training_num_uniques))
	plt.plot(x_axis, training_nums_prom_hist[0], label='Not Promoted')
	plt.plot(x_axis, training_nums_prom_hist[1], label='Promoted')
	plt.xticks(x_axis, training_num_uniques)
	plt.xlabel('Trainings Number')
	plt.ylabel('Count')
	plt.title('Number of Promoted associated with each Number of Trainings')
	plt.legend()
	plt.tight_layout()
	plt.savefig('visualizations/training_num_prom_barplot')
	plt.clf()
	# ----------------------------------------------------------------------------------------------
	# ----------------------------------------------------------------------------------------------
	prev_rating_uniques = sorted(df.previous_year_rating.unique())
	prev_rating_prom_corr = df_manipulated.previous_year_rating.corr(df.is_promoted)
	print("\nCorrelation between Previous Year Rating and Promotion is", prev_rating_prom_corr)

	prev_ratings = df[['previous_year_rating']].values.flatten()

	prev_ratings_prom_hist = np.zeros((2, len(prev_rating_uniques)))
	for i in range(train_size):
		j = np.where(prev_rating_uniques == prev_ratings[i])
		prev_ratings_prom_hist[prom[i], j] += 1

	x_axis = np.arange(len(prev_rating_uniques))
	plt.plot(x_axis, prev_ratings_prom_hist[0], label='Not Promoted')
	plt.plot(x_axis, prev_ratings_prom_hist[1], label='Promoted')
	plt.xticks(x_axis, prev_rating_uniques)
	plt.xlabel('Previous Year Rating')
	plt.ylabel('Count')
	plt.title('Number of Promoted associated with each Previous Year Ratings')
	plt.legend()
	plt.tight_layout()
	plt.savefig('visualizations/prev_rating_prom_barplot')
	plt.clf()
	# ----------------------------------------------------------------------------------------------
	# ----------------------------------------------------------------------------------------------
	service_len_uniques = sorted(df.length_of_service.unique())
	service_len_prom_corr = df_manipulated.previous_year_rating.corr(df.is_promoted)
	print("\nCorrelation between Length of Service and Promotion is", service_len_prom_corr)

	service_len = df[['length_of_service']].values.flatten()
	max_len = service_len.max()
	min_len = service_len.min()

	print('\nMaximum Service Length is', max_len)
	print('\nMinimum Service Length is', min_len)


	service_len_prom_hist = np.zeros((2, 6))
	for i in range(train_size):
		index = int((service_len[i] - 1) / 6)
		index = min(index, 5)
		service_len_prom_hist[prom[i], index] += 1


	labels = ['1-6', '7-12', '13-18', '19-24', '25-30', '31-37']

	x_axis = np.arange(6)
	plt.bar(x_axis, service_len_prom_hist[0], 0.4, label='Not Promoted')
	plt.bar(x_axis, service_len_prom_hist[1], 0.4, label='Promoted')
	plt.xticks(x_axis, labels)
	plt.xlabel('Length of Service')
	plt.ylabel('Historgram')
	plt.title('Number of Promoted associated with Length of Service')
	plt.legend()
	plt.tight_layout()
	plt.savefig('visualizations/service_len_prom_barplot')
	plt.clf()
	# ----------------------------------------------------------------------------------------------
	# ----------------------------------------------------------------------------------------------
	KPIs_classes = np.array([0, 1])
	KPI_prom_corr = df_manipulated['KPIs_met >80%'].corr(df.is_promoted)
	print("\nCorrelation between KPI Metric > 80 percent and Promotion is", KPI_prom_corr)

	KPIs = df[['KPIs_met >80%']].values.flatten()

	KPIs_prom_hist = np.zeros((2, 2))
	for i in range(train_size):
		j = np.where(KPIs_classes == KPIs[i])
		KPIs_prom_hist[prom[i], j] += 1

	x_axis = np.arange(2)
	plt.bar(x_axis, KPIs_prom_hist[0], 0.4, label='Not Promoted')
	plt.bar(x_axis, KPIs_prom_hist[1], 0.4, label='Promoted')
	plt.xticks(x_axis, ['Below 80%', 'Above 80%'])
	plt.xlabel('KPI Metric value')
	plt.ylabel('Historgram')
	plt.title('Number of Promoted associated with KPI Metric > 80%')
	plt.legend()
	plt.tight_layout()
	plt.savefig('visualizations/KPI_prom_barplot')
	plt.clf()
	# ----------------------------------------------------------------------------------------------
	# ----------------------------------------------------------------------------------------------
	rec_prom_corr = df_manipulated.recruitment_channel.corr(df.is_promoted)
	print("\nCorrelation between Recruitment Channel and promotion is", rec_prom_corr)

	rec = df[['recruitment_channel']].values.flatten()
	rec_prom_hist = np.zeros((2, len(rec_uniques)))
	for i in range(train_size):
		j = np.where(rec_uniques == rec[i])
		rec_prom_hist[prom[i], j] += 1

	x_axis = np.arange(len(rec_uniques))
	plt.bar(x_axis - 0.2, rec_prom_hist[0], 0.4, label='Not Promoted')
	plt.bar(x_axis + 0.2, rec_prom_hist[1], 0.4, label='Promoted')
	plt.xticks(x_axis, rec_uniques)
	plt.xlabel('Recruitment Channel')
	plt.ylabel('Histogram')
	plt.title('Number of promoted through each Recruitment Channel')
	plt.legend()
	plt.savefig('visualizations/rec_prom_barplot')
	plt.clf()
	# ----------------------------------------------------------------------------------------------
	# ----------------------------------------------------------------------------------------------
	reg_prom_corr = df_manipulated.region.corr(df.is_promoted)
	print("\nCorrelation between Region and promotion is", rec_prom_corr)

	reg = df[['region']].values.flatten()
	reg_prom_hist = np.zeros((2, len(reg_uniques)))
	for i in range(train_size):
		j = np.where(reg_uniques == reg[i])
		reg_prom_hist[prom[i], j] += 1

	x_axis = np.arange(len(reg_uniques))
	plt.bar(x_axis - 0.2, reg_prom_hist[0], 0.4, label='Not Promoted')
	plt.bar(x_axis + 0.2, reg_prom_hist[1], 0.4, label='Promoted')
	plt.xticks(x_axis, reg_uniques)
	plt.xticks(rotation=90)
	plt.xlabel('Region')
	plt.ylabel('Histogram')
	plt.title('Number of promoted in each region')
	plt.legend()
	plt.tight_layout()
	plt.savefig('visualizations/reg_prom_barplot')
	plt.clf()
##########################################################################
############################ CLASSIFICATION ##############################
##########################################################################
def modify_dataset():
	# double values of important variables
	df_manipulated['KPIs_met >80%'] = df_manipulated['KPIs_met >80%'].apply(lambda x: x*3)
	df_test['KPIs_met >80%'] = df_test['KPIs_met >80%'].apply(lambda x: x*3)

	df_manipulated['education'] = df_manipulated['education'].apply(lambda x: x*3)
	df_test['education'] = df_test['education'].apply(lambda x: x*3)

	df_manipulated['avg_training_score'] = df_manipulated['avg_training_score'].apply(lambda x: x*3)
	df_test['avg_training_score'] = df_test['avg_training_score'].apply(lambda x: x*3)
	# NOT GOING TO USE WITH DT
	'''
	# drop columns not really affecting promotions
	df_manipulated.drop(['gender', 'awards_won?', 'region'], axis=1, inplace=True)
	df_test.drop(['gender', 'awards_won?', 'region'], axis=1, inplace=True)
	# multiply promoted data
	promoted = df_manipulated.iloc[promoted_indices, :]
	df_manipulated.append(promoted, ignore_index=True)
	np.append(train_pred, np.repeat(1, len(promoted_indices)))
	'''


def model(X, Y, X_test, model_type='knn', criterion='gini', N=3):
	if model_type == 'knn':
		knn = KNN(n_neighbors = N)
		knn.fit(X, Y)
		model_pred = knn.predict(X_test)

	elif model_type == 'svm':
		svc = svm.SVC()
		svc.fit(X, Y)
		model_pred = np.array(svc.predict(X_test))

	elif model_type == 'dt':
		dtc = DT(criterion=criterion)
		dtc.fit(X, Y)
		model_pred = dtc.predict(X_test)

	return model_pred


def calc_accuracy(predictions, real):
	correct_count = 0
	for i in range(len(real)):
		if predictions[i] == real[i]:
			correct_count += 1
	
	return np.around(correct_count / len(real) * 100, 2)


def calc_F1(predictions, real):
	TP = 0
	FP = 0
	FN = 0
	for i in range(len(real)):
		if predictions[i] == real[i]:
			if predictions[i] == 1:
				TP += 1
		else:
			if predictions[i] == 0:
				FN += 1
			else:
				FP += 1

	return np.around(TP / (TP + 0.5 * (FP+FN)), 2)


run_data_visualizations()
modify_dataset()
preds = model(df_manipulated, train_pred, df_test, model_type='dt', criterion='entropy')
acc = calc_accuracy(preds, test_real_pred)
f1 = calc_F1(preds, test_real_pred)
print("\nAccuracy of test dataset predictions is ", acc, '%', sep='')
print("F1 score of test dataset predictions is", f1, '\n')
