# CropMaxify

Farmers in our country face many issues, such as scarce resources, lack of knowledge and unpredictable climate factor. To help them make the best use of their land and resources, we built a Deep Learning framework to predict the crop they should grow on their land. By taking inputs, namely, NPK content of fertilizers, temperature, humidity percentage, pH of the soil and rainfall (in mm), we can predict which crop they should grow.

The crops used as labels are namely Apple, Banana, Blackgram, Chickpea, Coconut, Coffee, Cotton, Grapes, Jute, Kidney beans, Lentil, Maize, Mango, Mothbeans, Mungbean, Muskmelon, Orange, Papaya, Pigeon, Peas, Pomegranate, Rice and Watermelon. Hence a total of 22 labels.

To do so, we created a 3-layer Neural Network with the help of PyTorch. We standardized and split the data in 80:20 for Training and Testing using SkLearn. After tuning the hyperparameters, we got an accuracy of 95.9% on the testing data.

After obtaining the desirable accuracy, with the help of Pickle, we extracted the PyTorch model along with the SkLearn Scaler and loaded it in the Streamlit app.py file. Getting the model on the website was the most challenging part since we needed to optimize the app.py file, as it took too much time to compile and showed many errors. To fix it, we reduced libraries in the app.py file, used only the necessary ones, and reduced the requirements for Streamlit.
The compatibility of libraries was also an issue that was taken care of.

*This project was created for the competition MasterStack held under Concetto 2022, IIT (ISM) Dhanbad.


