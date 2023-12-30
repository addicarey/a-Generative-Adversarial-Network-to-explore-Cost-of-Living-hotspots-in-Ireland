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
variables = {}
variable_names = ["Variable1","Variable2","Variable3","Variable4","Variable5"]

# Loop to take input for each variable
for name in variable_names:
    value = (input(f"Enter a numeric value for {name}: "))  # Convert input to float
    variables[name] = value

# Example formula: Calculate the average of the variables


Search_town = input(" Please enter a town:    ") 
def generate_realD_from_csv(data):
    # Select the columns corresponding to the 5 factors
    
    factors = data[[variables["Variable1"], variables["Variable2"], variables["Variable3"], variables["Variable4"], variables["Variable5"]]].values

    # Calculate the cost of living (if needed)
    # This is an example, modify it according to your needs
    costOL = np.sum(factors, axis=1) + np.random.normal(0, 0.1, len(factors))

    return factors, costOL
 # Replace with the name of the town you're looking for

# Check if the town exists in the DataFrame
if Search_town in data['town'].values:
    # Find the row corresponding to the town
    row = data[data['town'] == Search_town].iloc[0]

    # Extract longitude and latitude
    Long1 = row['longitude']
    Lat1 = row['latitude']
# Usage example
factors, costOL = generate_realD_from_csv(data)
Long1=Long1
Lat1=Lat1


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

def train_gan(generator, discriminator, gan, data, epochs, batchS, latentDim):
    for epoch in range(epochs):
        for _ in range(batchS):
            # Generate a batch of real data
            realFactors, real_costOL = generate_realD_from_csv(data)
            realLabels = np.ones(len(real_costOL))  # Label for real data

            # Generate a batch of fake data
            fakeCostOL = generator.predict(np.random.rand(batchS, latentDim))
            fakeLabels = np.zeros(len(fakeCostOL))  # Label for fake data

            # Train the discriminator
            dLossReal = discriminator.train_on_batch(real_costOL.reshape(-1, 1), realLabels)
            dLossFake = discriminator.train_on_batch(fakeCostOL, fakeLabels)

            # Calculate total discriminator loss
            dLoss = 0.5 * np.add(dLossReal, dLossFake)

            # Train the generator (through the GAN model)
            noise = np.random.rand(batchS, latentDim)
            fakeLabels_gan = np.ones(batchS)
            gLoss = gan.train_on_batch(noise, fakeLabels_gan)

        print(f"Epoch {epoch+1}, Discriminator Loss: {dLoss}, Generator Loss: {gLoss}")


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


realD = generate_realD_from_csv(data)

# Train the GAN
train_gan(generator, discriminator, gan, data, epochs, batchS, latentDim)



calculate_factor_importance(generator, latentDim) # calculates factor importance AC
import matplotlib.pyplot as plt

def calculate_factor_importance_and_plot(generator, latentDim, data):
    factorNames = ['rent', 'fuel', 'energy', 'food', 'education']
    factorImportance = []

    for i in range(len(factorNames)):
        # Create a baseline input (noise)
        baselineNoise = np.random.rand(1000, latentDim)
        
        # Create a modified input with the current factor set to its mean value
        modifiedNoise = baselineNoise.copy()
        modifiedNoise[:, i] = np.mean(data[factorNames[i]])
        
        # Generate output with baseline and modified input
        baselineOutput = generator.predict(baselineNoise).flatten()
        modifiedOutput = generator.predict(modifiedNoise).flatten()
        
        # Calculate the importance as the mean absolute difference
        importance = np.mean(np.abs(baselineOutput - modifiedOutput))
        factorImportance.append(importance)
    

    return factorImportance

# Example usage
factorImportance = calculate_factor_importance_and_plot(generator, latentDim, data)

# generates synthetic data AC
numSamp = 1000
generated_costOL = generator.predict(np.random.rand(numSamp, latentDim))
'''
hMap = folium.Map(location=[Long1, Lat1], zoom_start=12) # creates a map centered around a location AC


# adds a heatmap layer using data AC
heatData = [[row['latitude'], row['longitude']] for index, row in data.iterrows()]
folium.plugins.HeatMap(heatData).add_to(hMap)

# displays the map AC
hMap.save('costOL_heatmap.html')
# visualizes generated data AC
fig, axs = plt.subplots(2, 1, figsize=(10, 12))'''
hMap = folium.Map(location=[Lat1, Long1], zoom_start=12)

# Add a marker to the map for verification
folium.Marker([Lat1, Long1], popup="Your Location").add_to(hMap)

# Explicitly set heatmap data (replace with actual coordinates for testing)
heatData = [[Lat1, Long1]]

# Add the heatmap layer
folium.plugins.HeatMap(heatData).add_to(hMap)

# Save the heatmap to an HTML file
hMap.save('costOL_heatmap.html')
fig, axs = plt.subplots(2, 1, figsize=(10, 12))

# First plot (bar graph for factor importance)
axs[0].bar(['rent', 'fuel', 'energy', 'food', 'education'], factorImportance, color='skyblue')
axs[0].set_xlabel('Factors')
axs[0].set_ylabel('Importance')
axs[0].set_title('Importance of Each Factor on Cost of Living')

# Second plot (scatter plot for generated data)
numSamp = 1000
generated_costOL = generator.predict(np.random.rand(numSamp, latentDim))
sum_of_factors = np.sum(np.random.rand(numSamp, 5), axis=1)
axs[1].scatter(sum_of_factors, generated_costOL, label='Generated Data')
axs[1].set_xlabel('Sum of Factors')
axs[1].set_ylabel('Cost of Living')
axs[1].legend()

plt.tight_layout()
plt.show()



