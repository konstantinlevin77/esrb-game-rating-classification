# let's define a function that load the model, predict the data and return the results.
def deployFunction(data):
    
    model = pickle.load(open("randomforest.pkl",mode="rb"))
    int2data = json.load(open("int2data.json))
    pred = model.predict([data])[0]
    result = int2data[pred]
    return result


deployFunction(x_test[321])


int2data[y_test[321]]
