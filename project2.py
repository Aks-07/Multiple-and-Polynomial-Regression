
# In[] importing library and packages

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


# In[] reading the data

home = pd.read_csv('home_price.csv')

# In[] check the number of records and columns in the data

home.columns
home.shape
home.head(5)

# In[] drop the id and date columns since they seem irrevelant on inspection

home.drop('id',inplace = True,axis = 1)
home.drop('date',inplace = True,axis = 1)

home.shape

# In[] check for nulls in the data and the general statistics 
 
print(home.isnull().any())

home.describe()

# no nulls in the data and the data seems fairly clean, so no further cleaning required

# In[]: 
    
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
g = sns.PairGrid(home)
g = g.map(plt.scatter)
plt.show()

# In[]

##################################### Part 1
     
# In[] checking for importance of features by recursive feature elimination 

from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

# we start of by selecting all the 18 features and rank them as per their importance

feature_home = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
                'waterfront', 'view', 'condition', 'grade', 'sqft_above',
                'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long',
                'sqft_living15', 'sqft_lot15']
x_home = home[feature_home]
y_home = home['price']

#use linear regression as the model

lr = LinearRegression()

#rank all features, i.e continue the elimination until the last one

rfe = RFE(lr, n_features_to_select=1)
rfe.fit(x_home,y_home)
 
print("Features sorted by their rank:")
print(sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), x_home)))


# we can use these features in the order of the ranks to check for different runs of regression and check 
# which gives the higher value of r-square for test and training sets
# we can start of with 1 feature and keep adding each feature in the order of ranks or keep all features
# and remove them in order of lowest ranks

# In this case, we observed that highest value of r-squared was for the first 12 features raknwise and 
# it was he same even if all 18 features are kept. So for now, we run the multiple regression model 
# keeping all the features


# In[] running regression in scikit learn

import statsmodels.api as sm
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

x_train, x_test, y_train, y_test = train_test_split(x_home, y_home, random_state = 0)

linreg = LinearRegression().fit(x_train, y_train)
regression_model = LinearRegression()
regression_model.fit(x_train, y_train)

# checking the regression coefficients

print('linear model coeff (w): {}'.format(linreg.coef_))
print('linear model intercept (b): {:.3f}' .format(linreg.intercept_))

# checking the r-squared values for the model for training and test data

print('R-squared score (training): {:.3f}' .format(linreg.score(x_train, y_train)))
print('R-squared score (test): {:.3f}'     .format(linreg.score(x_test, y_test)))

home_y_pred = linreg.predict(x_test)

# calculating the residual sum of squares error

print("Mean squared error: %.2f"  % mean_squared_error(y_test, home_y_pred))


# In[] running regression in statsmodels - just for reference

model = sm.OLS(y_train, x_train).fit()
model.summary()


# In[] 

####################################################    Part 2

# In[]

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

print('\nNow we transform the original input data to add\npolynomial features up to degree 2 (quadratic)\n')
poly = PolynomialFeatures(degree=2)
x_F1_poly = poly.fit_transform(x_home)

x_train1, x_test1, y_train1, y_test1 = train_test_split(x_F1_poly, y_home,random_state = 0)

linreg = LinearRegression().fit(x_train1, y_train1)
home_y_pred1 = linreg.predict(x_test1)

print('(poly deg 2) linear model coeff (w):\n{}' .format(linreg.coef_))
print('(poly deg 2) linear model intercept (b): {:.3f}'.format(linreg.intercept_))
print('(poly deg 2) R-squared score (training): {:.3f}'.format(linreg.score(x_train1, y_train1)))
print('(poly deg 2) R-squared score (test): {:.3f}\n'.format(linreg.score(x_test1, y_test1)))


# In[]: this part is also not required i guess???????????????????????

#plt.scatter(x_test, y_test,  color='black')

#plt.plot(x_test['sqft_above'],y_test,'.',
 #        x_test['sqft_above'],home_y_pred,'-')

    
plt.scatter(x_test['sqft_above'], y_test,  color='black')
plt.plot(x_test['sqft_above'],home_y_pred1,'-')
       
plt.title('Polynomial Regression')
plt.xlabel('Feature value (x)')
plt.ylabel('Target value (y)')    
plt.show()


# In[]:  what is this for??????????????????????????????????

import seaborn as sns; sns.set(color_codes=True)
data358 = pd.concat([x_test['sqft_above'],y_test],axis = 1)
ax = sns.regplot(x=x_test['sqft_above'], y=y_test, data=data358)
plt.show()


# In[]: why only a few variables and why not all???????????????????????????????????????????///

# visualizing polynomial regressions

fig, axs = plt.subplots(2, 2, figsize=(15, 15))
axs[0, 0].plot(x_test['sqft_above'],y_test,'.', x_test['sqft_above'],home_y_pred,'-')
axs[0, 0].set_title('Polynomial Regression')
axs[0, 0].set_xlabel('Feature value (x)')
axs[0, 0].set_ylabel('Target value (y)')    


axs[1, 0].plot(x_test['sqft_basement'],y_test,'.', x_test['sqft_basement'],home_y_pred,'-')
axs[1, 0].set_title('Polynomial Regression')
axs[1, 0].set_xlabel('Feature value (x)')
axs[1, 0].set_ylabel('Target value (y)')    


axs[0, 1].plot(x_test['lat'],y_test,'.', x_test['lat'],home_y_pred,'-')

axs[0, 1].set_title('Polynomial Regression')
axs[0, 1].set_xlabel('Feature value (x)')
axs[0, 1].set_ylabel('Target value (y)')    


axs[1, 1].plot(x_test['long'],y_test,'.', x_test['long'],home_y_pred,'-')

axs[1, 1].set_title('Polynomial Regression')
axs[1, 1].set_xlabel('Feature value (x)')
axs[1, 1].set_ylabel('Target value (y)')    
plt.show()
plt.tight_layout


# In[]:  what is this for???????????????????????????????????????????????????????????????????

import seaborn as sns; sns.set(color_codes=True)
data358 = pd.concat([x_test['sqft_above'],x_test['sqft_basement'],x_test['lat'],x_test['long'],y_test],axis = 1)
#ax = sns.regplot(x=x_test['sqft_above'], y=y_test, data=data358)
#plt.show()

#sns.regplot(x=x_test['sqft_above'], y=y_test, data=data358)
tidy = (
    data358.stack() # pull the columns into row variables   
      .to_frame() # convert the resulting Series to a DataFrame
      .reset_index() # pull the resulting MultiIndex into the columns
      .rename(columns={0: 'val'}) # rename the unnamed column
)
fig, (ax1, ax2) = plt.subplots(ncols=2 ,squeeze=True)
sns.regplot(x=x_test['sqft_above'],y= y_test, data=tidy,ax = ax1)
sns.regplot(x=x_test['sqft_basement'],y= y_test, data=tidy, ax=ax2)
fig, (ax3, ax4) = plt.subplots(ncols=2 )
sns.regplot(x=x_test['lat'],y= y_test, data=tidy,ax = ax3)
sns.regplot(x=x_test['long'],y= y_test, data=tidy, ax=ax4)
plt.tight_layout()
#sns.regplot(x=x_test['lat'],y= y_test, ax=ax3)
#sns.regplot(x=x_test['long'], y=y_test, ax=ax4)
plt.show()


# In[] 

# using different features (subsets) to visualize polynomial regressions




# In[]: using testing and training sets to select a polynomial degree n 

print('\nNow we transform the original input data to add\npolynomial features up to degree n \n')


train3 = []
test3 = []
train5 = []
test5 = []
train8 = []
test8 = []

for n in range(1,5):
    poly = PolynomialFeatures(degree=n)
    x_F1_poly1 = poly.fit_transform(x_home)

    x_train3, x_test3, y_train3, y_test3 = train_test_split(x_F1_poly1, y_home,test_size = 0.2,random_state = 0)
    x_train5, x_test5, y_train5, y_test5 = train_test_split(x_F1_poly1, y_home,test_size = 0.25,random_state = 0)
    x_train8, x_test8, y_train8, y_test8 = train_test_split(x_F1_poly1, y_home,test_size = 0.3,random_state = 0)
    linreg3 = LinearRegression().fit(x_train3, y_train3)
    linreg5= LinearRegression().fit(x_train5, y_train5)
    linreg8 = LinearRegression().fit(x_train8, y_train8)
    train3.append(linreg3.score(x_train3, y_train3))
    test3.append(linreg3.score(x_test3, y_test3))
    train5.append(linreg5.score(x_train5, y_train5))
    test5.append(linreg5.score(x_test5, y_test5))
    train8.append(linreg8.score(x_train8, y_train8))
    test8.append(linreg8.score(x_test8, y_test8))
    

for i in range(0,4):    
    print('train score for {} degree polynnomial - 80 20 ratio dataset is {}'.format((i+1),train3[i]))
    print('test score for {} degree polynnomial - 80 20 ratio dataset is {}'.format((i+1),test3[i]))   

print('\n')
for i in range(0,4):    
    print('train score for {} degree polynnomial - 75 25 ratio dataset is {}'.format(i+1,train5[i]))
    print('test score for {} degree polynnomial - 75 25 ratio dataset is {}'.format(i+1,test5[i]))

print('\n')
for i in range(0,4):    
    print('train score for {} degree polynnomial - 70 20 ratio dataset is {}'.format(i+1,train8[i]))
    print('test score for {} degree polynnomial - 70 20 ratio dataset is {}'.format(i+1,test8[i]))

#plt.scatter(range(1,5), test3, c ='g')
#plt.scatter(range(1,50), test5, c ='b')
#plt.scatter(range(1,50), test8, c ='b')
#plt.show()
            
#print('(poly deg 2) linear model coeff (w):\n{}'.format(linreg.coef_))
#print('(poly deg 2) linear model intercept (b): {:.3f}'.format(linreg.intercept_))
#print('(poly deg 2) R-squared score (training): {:.3f}'.format(linreg.score(x_train2, y_train2)))
#print('(poly deg 2) R-squared score (test): {:.3f}\n'.format(linreg.score(x_test2, y_test2)))

plt.scatter(range(1,5), test3, c='r')
plt.scatter(range(1,5), test5, c='b')
plt.scatter(range(1,5), test8, c='m')
plt.show()


# In[ ]:

#################################################### Part 3

# In[]: using polynomial of degree 2 from the above results 

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures


poly = PolynomialFeatures(degree=2)
x_F24_poly = poly.fit_transform(x_home)
x_train24, x_test24, y_train24, y_test24 = train_test_split(x_F24_poly, y_home, random_state = 0)
linreg = LinearRegression().fit(x_train24, y_train24)

print('linear model coeff (w): {}' .format(linreg.coef_))
print('linear model intercept (b): {:.3f}' .format(linreg.intercept_))
print('R-squared score (training): {:.3f}' .format(linreg.score(x_train24, y_train24)))
print('R-squared score (test): {:.3f}' .format(linreg.score(x_test24, y_test24)))


## apply ridge penalty to the regression result

print('\nNow we transform the original input data to add\npolynomial features up to degree 2 (quadratic) and also apply ridge\n')
poly = PolynomialFeatures(degree=2)
X_F25_poly = poly.fit_transform(x_home)


x_train25, x_test25, y_train25, y_test25 = train_test_split(X_F25_poly, y_home,random_state = 0)

x_train_scaled25 = scaler.fit_transform(x_train25)
x_test_scaled25 = scaler.transform(x_test25)

linridge = Ridge().fit(x_train_scaled, y_train25)

print('Home dataset')   
print('ridge regression linear model intercept: {}'.format(linridge.intercept_))
print('ridge regression linear model coeff:\n{}'.format(linridge.coef_))
print('R-squared score (training): {:.3f}'.format(linridge.score(x_train_scaled25, y_train25)))
print('R-squared score (test): {:.3f}' .format(linridge.score(x_test_scaled25, y_test25)))


#print('Number of non-zero features: {}' .format(np.sum(linridge.coef_ != 0)))


# In[]: changing the alpha parameter to select the best penalty

print('Ridge regression: effect of alpha regularization parameter\n')
for this_alpha in [0, 1, 10, 20, 50, 100, 1000]:
    linridge = Ridge(alpha = this_alpha).fit(x_train_scaled25, y_train25)
    r2_train = linridge.score(x_train_scaled25, y_train25)
    r2_test = linridge.score(x_test_scaled25, y_test25)
    num_coeff_bigger = np.sum(abs(linridge.coef_) > 1.0)
    print('Alpha = {:.2f}\nnum abs(coeff) > 1.0: {}, r-squared training: {:.2f}, r-squared test: {:.2f}\n'
         .format(this_alpha, num_coeff_bigger, r2_train, r2_test))


# In[]: selecting polynomial of degree 2

from sklearn.linear_model import Lasso
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

poly = PolynomialFeatures(degree=2)
x_F49_poly = poly.fit_transform(x_home)

x_train50, x_test50, y_train50, y_test50 = train_test_split(x_F49_poly, y_home, random_state = 0)

x_train_scaled51 = scaler.fit_transform(x_train50)
x_test_scaled51 = scaler.transform(x_test50)


## applying lasso penalty to the regression result

linlasso = Lasso(alpha=2.0, max_iter = 10000).fit(x_train_scaled51, y_train50)

print('Home dataset')
print('lasso regression linear model intercept: {}'.format(linlasso.intercept_))
print('lasso regression linear model coeff:\n{}'.format(linlasso.coef_))
print('Non-zero features: {}'.format(np.sum(linlasso.coef_ != 0)))
print('R-squared score (training): {:.3f}'.format(linlasso.score(x_train_scaled51, y_train50)))
print('R-squared score (test): {:.3f}\n'.format(linlasso.score(x_test_scaled51, y_test50)))
print('Features with non-zero weight (sorted by absolute magnitude):')

for e in sorted (list(zip(list(x_home), linlasso.coef_)),
                key = lambda e: -abs(e[1])):
    if e[1] != 0:
        print('\t{}, {:.3f}'.format(e[0], e[1]))


# In[]: changing the alpha parameter to select the best penalty

print('Lasso regression: effect of alpha regularization\nparameter on number of features kept in final model\n')

for alpha in [0.5, 1, 2, 3, 5, 10, 20, 50]:
    linlasso = Lasso(alpha, max_iter = 10000).fit(x_train_scaled51, y_train50)
    r2_train = linlasso.score(x_train_scaled51, y_train50)
    r2_test = linlasso.score(x_test_scaled51, y_test50)
    
    print('Alpha = {:.2f}\nFeatures kept: {}, r-squared training: {:.2f}, r-squared test: {:.2f}\n'
         .format(alpha, np.sum(linlasso.coef_ != 0), r2_train, r2_test))


# In[] 

############################################################# # Part 4

# In[]  implementing knn regression


from sklearn.neighbors import KNeighborsRegressor

x_train80, x_test80, y_train80, y_test80 = train_test_split(x_home, y_home, random_state = 0)

knnreg = KNeighborsRegressor(n_neighbors = 5).fit(x_train80, y_train80)

print(knnreg.predict(x_test80))

print('R-squared test score: {:.3f}' .format(knnreg.score(x_test80, y_test80)))


# In[]: selecting the best value of k

# plot k-NN regression on sample dataset for different values of K
fig, subaxes = plt.subplots(5, 1, figsize=(5,20))
#X_predict_input = np.linspace(-3, 3, 500).reshape(-1,1)
X_train98, X_test98, y_train98, y_test98 = train_test_split(x_home, y_home,test_size = 0.20, random_state = 0)
X_train99, X_test99, y_train99, y_test99 = train_test_split(x_home, y_home,test_size = 0.25, random_state = 0)
X_train100, X_test100, y_train100, y_test100 = train_test_split(x_home, y_home,test_size = 0.30, random_state = 0)

train_score98 = []
train_score99 = []
train_score100 = []
test_score98 = []
test_score99 = []
test_score100 = []


#### why only these k values?????????????????????????????????????????????????????????????????????

for K in [1, 3, 7, 15, 55]:
    knnreg98 = KNeighborsRegressor(n_neighbors = K).fit(X_train98, y_train98)
    train_score98.append(knnreg98.score(X_train98, y_train98))
    test_score98.append(knnreg98.score(X_test98, y_test98))
    knnreg99 = KNeighborsRegressor(n_neighbors = K).fit(X_train99, y_train99)
    train_score99.append(knnreg99.score(X_train99, y_train99))
    test_score99.append(knnreg99.score(X_test99, y_test99))
    knnreg100 = KNeighborsRegressor(n_neighbors = K).fit(X_train100, y_train100)
    train_score100.append(knnreg100.score(X_train100, y_train100))
    test_score100.append(knnreg100.score(X_test100, y_test100))
    
i=0
print('80 20 dataset')
for K in [1, 3, 7, 15, 55]:
    print(('KNN Regression (K={})\nTrain R^2 = {:.3f},  Test R^2 = {:.3f}'.format(K, train_score98[i], test_score98[i])))
    i = i+1
    
print('\n75 25 dataset')
j=0
for K in [1, 3, 7, 15, 55]:
    print(('KNN Regression (K={})\nTrain R^2 = {:.3f},  Test R^2 = {:.3f}'.format(K, train_score99[j], test_score99[j])))
    j = j+1
print('\n70 30 dataset')
z=0
for K in [1, 3, 7, 15, 55]:
    print(('KNN Regression (K={})\nTrain R^2 = {:.3f},  Test R^2 = {:.3f}'.format(K, train_score100[z], test_score100[z])))
    z = z+1          
    
    
    #thisaxis.plot(y_test, y_predict_output)
    #thisaxis.plot(X_train, y_train, 'o', alpha=0.9, label='Train')
    #thisaxis.plot(X_test, y_test, '^', alpha=0.9, label='Test')
    #thisaxis.set_xlabel('Input feature')
    #thisaxis.set_ylabel('Target value')
    #thisaxis.set_title('KNN Regression (K={})\n\
#Train $R^2 = {:.3f}$,  Test $R^2 = {:.3f}$'
#                      .format(K, train_score, test_score))
   # thisaxis.legend()
    #plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    #plt.show()
    
# In[]
