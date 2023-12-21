# required packages AC
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import folium
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from folium.plugins import HeatMap

data = pd.read_csv('your_spatial_data.csv') # loads and preprocess data AC

def generate_realD(samp): # generates synthetic data AC
    factors = np.random.rand(samp, 5)  # 5 factors affecting cost of living AC
    costOL = np.sum(factors, axis=1) + np.random.normal(0, 0.1, samp)
    return factors, costOL

def build_generator(latentDim): # generator model AC
    model = Sequential()
    model.add(Dense(16, input_dim=latentDim, activation='relu'))
    model.add(Dense(1, activation='linear'))
    return model

def build_discriminator(): # discriminator model AC
    model = Sequential()
    model.add(Dense(16, input_dim=1, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    return model

def build_gan(generator, discriminator): # builds and compiles model AC
    discriminator.trainable = False
    ganInput = Input(shape=(latentDim,))
    generatedData = generator(ganInput)
    ganOutput = discriminator(generatedData)
    gan = Model(ganInput, ganOutput)
    gan.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001))
    return gan

def train_gan(generator, discriminator, gan, realD, epochs, batchS): # training loop AC
    for epoch in range(epochs):
        for _ in range(batchS):
            realFactors, real_costOL = realD
            realLabels = np.ones(batchS)  # all real data is labeled as 1? AC
            fakeCostOL = generator.predict(np.random.rand(batchS, latentDim))
            fakeLabels = np.zeros(batchS)
            
            dLossReal = discriminator.train_on_batch(real_costOL, realLabels)
            dLossFake = discriminator.train_on_batch(fakeCostOL, fakeLabels)
            
            dLoss = 0.5 * np.add(dLossReal, dLossFake)
            
            fakeLabels_gan = np.ones(batchS)
            gLoss = gan.train_on_batch(np.random.rand(batchS, latentDim), fakeLabels_gan)
            
        print(f"Epoch {epoch+1}, Discrinator Loss: {dLoss}, Generator Loss: {gLoss}")

def calculate_factor_importance(generator, latentDim): # calculates factor importance AC
    factorEffects = np.zeros(latentDim) # initializes an array to store the effects of each factor AC
    noise = np.random.rand(1000, latentDim) # generates random noise AC
    generated_costOL = generator.predict(noise) # generates synthetic cost of living values AC
    meanEffect = np.mean(generated_costOL) # calculates the effect of each factor by subtracting the mean effect AC
    
    for i in range(latentDim): # calculates the effect of each individual factor AC
        factorNoise = noise.copy()
        factorNoise[:, i] = 0  # sets one factor to zero while keeping others constant AC
        factorEffect = np.mean(generator.predict(factorNoise) - generated_costOL)
        factorEffects[i] = factorEffect
    
    
    print("Factor Effects:") # prints the factor effects AC
    for i, effect in enumerate(factorEffects):
        print(f"Factor {i+1}: {effect}")

# hyperparameters AC
latentDim = 10  # latent dimension for generator AC
epochs = 10 # epochs AC
batchS = 128 # batch size AC

# build models AC
generator = build_generator(latentDim)
discriminator = build_discriminator()

discriminator.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy']) # compiles the discriminator model AC

gan = build_gan(generator, discriminator) # builds GAN model and compiles it AC

# training loop AC
realD = generate_realD(batchS)
train_gan(generator, discriminator, gan, realD, epochs, batchS)


calculate_factor_importance(generator, latentDim) # calculates factor importance AC

# generates synthetic data AC
numSamp = 1000
generated_costOL = generator.predict(np.random.rand(numSamp, latentDim))

hMap = folium.Map(location=[53.350140, -6.266155], zoom_start=12) # creates a map centered around a location AC


# adds a heatmap layer using data AC
heatData = [[row['latitude'], row['longitude']] for index, row in data.iterrows()]
folium.plugins.HeatMap(heatData).add_to(hMap)

# displays the map AC
hMap.save('costOL_heatmap.html')
# visualizes generated data AC
plt.scatter(np.sum(np.random.rand(numSamp, 5), axis=1), generated_costOL, label='Generated Data')
plt.xlabel('Sum of Factors')
plt.ylabel('Cost of Living')
plt.legend()
plt.show()


