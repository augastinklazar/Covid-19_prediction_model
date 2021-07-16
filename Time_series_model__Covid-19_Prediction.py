#!/usr/bin/env python
# coding: utf-8

# In[1]:


from fbprophet import Prophet


# In[2]:


dir(Prophet)


# In[3]:


import pandas as pds


# In[4]:


df = pds.read_csv(r'D:\time series\covid_19_clean_complete.csv')     #data pre-processing


# In[5]:


df.head()


# In[12]:


df.shape


# In[6]:


df.dtypes


# In[8]:


df['Date'] = pds.to_datetime(df['Date'])  #convert to date format


# In[9]:


df.dtypes


# In[10]:


df.isnull().sum()    #check missing data


# In[11]:


df['Date'].nunique()    #unique no of dates


# In[15]:


total = df.groupby(['Date'])['Confirmed','Deaths','Recovered','Active'].sum().reset_index()


# In[16]:


total.head()   # data for predictions i.e, to build the time series model.


# In[ ]:


# ds is expected column name for date, y is that of data to be predicted...


# In[17]:


df_prophet = total.rename(columns = {'Date':'ds','Confirmed':'y'})   #df.prophet is the new data frame with col name updations.


# In[18]:


df_prophet.head()


# In[19]:


m = Prophet()   #initialising data


# In[21]:


model = m.fit(df_prophet)   #fitting the data


# In[ ]:





# In[22]:


model.seasonalities


# In[23]:


future_global = model.make_future_dataframe(periods = 30, freq = 'D')


# In[24]:


future_global.head()


# In[26]:


df_prophet['ds'].tail()    #note the last date


# In[27]:


future_global.tail()     #last date here is different (30 days more)[testing data]


# In[ ]:


# predictions on the future dates


# In[30]:


prediction = model.predict(future_global)
prediction


# In[32]:


prediction[['ds','yhat','yhat_lower','yhat_upper']].tail()    # predictions


# In[ ]:


#visualise predictions


# In[34]:


model.plot(prediction)   #light blue line shows the trend


# In[ ]:


# visualise by conditions


# In[35]:


model.plot_components(prediction)


# In[ ]:


# plotting trend changes


# In[36]:


from fbprophet.plot import add_changepoints_to_plot


# In[38]:


fig = model.plot(prediction)
add_changepoints_to_plot(fig.gca(), model, prediction)


# In[ ]:


#cross validate time serias data


# In[39]:


from fbprophet.diagnostics import cross_validation


# In[40]:


df_cv = cross_validation(model, horizon = '30 days', period = '15 days', initial = '90 days')


# In[41]:


df_cv.head()    #new col cutoff appears


# In[42]:


df_cv.shape


# In[ ]:


#obtain performance metrics for the model


# In[ ]:


# different errors that exists


# In[43]:


from fbprophet.diagnostics import performance_metrics


# In[45]:


df_performance = performance_metrics(df_cv)


# In[46]:


df_performance.head()


# In[ ]:


#mse - mean squared error
#rmse - root mean square error
#mae - mean absolute error
#mape - mean absolute percent error


# In[ ]:


# visualising rmse


# In[48]:


from fbprophet.plot import plot_cross_validation_metric
df_performance = plot_cross_validation_metric(df_cv, metric = 'rmse')


# In[49]:


df_performance = plot_cross_validation_metric(df_cv, metric = 'mse')


# In[50]:


df_performance = plot_cross_validation_metric(df_cv, metric = 'mape')


# In[ ]:




