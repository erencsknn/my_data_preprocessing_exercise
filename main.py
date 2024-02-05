# import pandas as pd
# import math
# from scipy.spatial import distance

# data_frame = pd.read_csv('adult.data')

# data_frame.columns = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status", "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss", "hours_per_week", "native_country","y" ]

# print(data_frame.head()) # first 5 rows



# ecuclidean distance

# human_1 = [30,4000]
# human_2 = [28,3500]
# human_3 = [59,12000]

# hh_euclidean_distance = distance.euclidean(human_1, human_2)
# hm_euclidean_distance = distance.euclidean(human_1, human_3)
# mh_euclidean_distance = distance.euclidean(human_2, human_3)
# print(f"Euclidean distance between human 1 and human 2 is {hh_euclidean_distance}, human 1 and human 3 is {hm_euclidean_distance}, human 2 and human 3 is {mh_euclidean_distance}")

# cosine distance
# a_vector= (3,4)
# b_vector = (4,2)
# a_b_cosine_distance = distance.cosine(a_vector, b_vector)
# a_b_similarity = 1 - a_b_cosine_distance
# print(f"Cosine distance between a and b is {a_b_cosine_distance}, similarity between a and b is {a_b_similarity}")

# jaccard distance
# human_1 = (1,0,1,0,0,0)
# human_2 = (1,1,0,0,0,0)
# human_3 = (1,0,1,0,1,0)

# distance_1_2 = 1-distance.jaccard(human_1, human_2)
# distance_1_3 = 1-distance.jaccard(human_1, human_3)
# distance_2_3 = 1-distance.jaccard(human_2, human_3)

# print(f"Jaccard distance between human 1 and human 2 is {distance_1_2}, human 1 and human 3 is {distance_1_3}, human 2 and human, human2 and human 3 is {distance_2_3}")

# missing data instances
# missing_data = pd.Series(data = np.array([None,np.nan,1,2,"merhaba"]))

#missing data detection

### isnull() missing is true
### isna() isnull() same function
### notnull() missing is false
# print(missing_data.isnull().sum())    # missing data count

# for i in missing_data.isnull():
#     if i == True:
#         print("missing data found")
# print(missing_data[missing_data.isnull()])

# missing data filling or dropping:
# data_frame = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")

# print(data_frame.isnull().sum())

# data_frame.fillna({"Age": -1, "Cabin": "Unknown", "Embarked": "Unknown"}).to_csv("titanic_missing.csv")
# data_frame.fillna(method="backfill").to_csv("titanic_missing_backfill.csv")
# print(data_frame.Age.interpolate(method= 'index').tail(10))
# print(data_frame.shape)
# print(data_frame.dropna().shape)
# print(data_frame.dropna(thresh=12).shape)
# print(data_frame.dropna(subset=["Age", "Cabin"]))


# outlier detection and handling

# print(data_frame.Age.isnull().sum())
# fill missing data
# data_frame.Age.fillna(value=data_frame.Age.mean(), inplace=True)


## IQR METHOD
# data_frame_age = data_frame.Age.copy()
# q_1 = data_frame_age.quantile(0.25)
# q_3 = data_frame_age.quantile(0.75)
# IQR = q_3 - q_1
# lower_bound = q_1 - 1.5*IQR
# upper_bound = q_3 + 1.5*IQR
# print(f"Lower bound: {lower_bound}, Upper bound: {upper_bound}")


# data_frame_age[data_frame_age>upper_bound] = upper_bound
# data_frame_age[data_frame_age<lower_bound] = lower_bound
# data_frame_age.plot.box()
# plt.show()

## Z-SCORE METHOD

# data_frame_age = data_frame.Age.copy()
# std = data_frame_age.std()
# mean = data_frame_age.mean()
# outlier_data = data_frame_age[data_frame_age> std*3] ## outlier data detection
# print(outlier_data)

# data_frame = pd.read_csv("https://raw.githubusercontent.com/erkansirin78/datasets/master/Purchase_Prediction_Data.csv")
# data_frame_age = data_frame.Age.copy()

## normalization
# data_frame_age.fillna(data_frame_age.mean(), inplace=True) # fill missing data

# normalized_age = (data_frame_age - data_frame_age.min()) / (data_frame_age.max() - data_frame_age.min()) # 0-1 normalization

# print(normalized_age)

## standardization
# standardized_age = (data_frame_age - data_frame_age.mean()) / data_frame_age.std() 
# print(standardized_age)

## discretization
# age_bins = [0, 20, 40, 100]
# discretization_ages =  pd.cut(x=data_frame_age,bins=age_bins, labels=["Young", "Middle", "Old"]) # discretization
# # print(discretization_ages)
# data_frame_purchase = data_frame.Purchased.copy()
# print(pd.get_dummies(data_frame_purchase,drop_first=True).head()) # one hot encoding
