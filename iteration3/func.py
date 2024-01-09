
import numpy as np
import os
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
import pandas as pd


def load_csv():
    data = pd.read_csv('Book2.csv')
    return data


# extracts variables from datafarme AC
def generate_realD_from_csv(data , variables):
    factor = [variables["Variable1"], variables["Variable2"], variables["Variable3"], variables["Variable4"], variables["Variable5"]]
    factors = data[factor]
    costOL = np.sum(factors, axis=1) + np.random.normal(0, 0.1, len(factors)) # calculates cost of living by adding factors with random noise AC
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

def build_gan(generator, discriminator,latentDim): # builds and compiles model AC
    discriminator.trainable = False
    ganInput = Input(shape=(latentDim,))
    generatedData = generator(ganInput)
    ganOutput = discriminator(generatedData)
    gan = Model(ganInput, ganOutput)
    gan.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001))
    return gan

def train_gan(generator, discriminator, gan, data, epochs, batchS, latentDim , variables):
    for epoch in range(epochs):
        for _ in range(batchS):
            # generates a batch of real data AC
            realFactors, real_costOL = generate_realD_from_csv(data,variables)
            realLabels = np.ones(len(real_costOL))  # label for real data

            # generates a batch of fake data AC
            fakeCostOL = generator.predict(np.random.rand(batchS, latentDim))
            fakeLabels = np.zeros(len(fakeCostOL))  # label for fake data AC
            real_costOL_array = real_costOL.values
            # trains the discriminator AC
            dLossReal = discriminator.train_on_batch(real_costOL_array.reshape(-1, 1), realLabels)
            dLossFake = discriminator.train_on_batch(fakeCostOL, fakeLabels)

            
            dLoss = 0.5 * np.add(dLossReal, dLossFake) # calculates total discriminator loss AC

            # trains the generator (through the GAN model) AC
            noise = np.random.rand(batchS, latentDim)
            fakeLabels_gan = np.ones(batchS)
            gLoss = gan.train_on_batch(noise, fakeLabels_gan)
        
        print(f"Epoch {epoch+1}, Discriminator Loss: {dLoss}, Generator Loss: {gLoss}") # prints epoch number, d loss and g loss AC


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

def calculate_factor_importance_and_plot(generator, latentDim, data):
    factorNames = ['Housing', 'Transport', 'Energy', 'Food', 'Education']
    factorImportance = []

    for i in range(len(factorNames)):
        
        baselineNoise = np.random.rand(1000, latentDim) # creates a baseline input (noise) AC
        
        # creates a modified input with the current factor set to its mean value AC
        modifiedNoise = baselineNoise.copy()
        modifiedNoise[:, i] = np.mean(data[factorNames[i]])
        
        # generates output with baseline and modified input AC
        baselineOutput = generator.predict(baselineNoise).flatten()
        modifiedOutput = generator.predict(modifiedNoise).flatten()
        
        # calculates the importance as the mean absolute difference AC
        importance = np.mean(np.abs(baselineOutput - modifiedOutput))
        factorImportance.append(importance)
    factorImportance = scale_to_range(factorImportance, -10, 10)
    

    return factorImportance

def scale_to_range(lst, new_min, new_max): # scales factors from -10,10 for bar chart AC
    minimum = min(lst)
    maximum = max(lst)
    return [new_min + ((x - minimum) * (new_max - new_min) / (maximum - minimum)) for x in lst]