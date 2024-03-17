top_data = data[['State','Total day minutes']]
top_data = top_data.groupby('State').sum()
top_data = top_data.sort_values('Total day minutes',ascending=False)
top_data = top_data[:3].index.values
sns.boxplot(y='State', 
            x='Total day minutes', 
            data=data[data.State.isin(top_data)], palette='Set3');