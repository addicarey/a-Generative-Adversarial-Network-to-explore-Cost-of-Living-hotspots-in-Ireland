import folium
from folium.plugins import HeatMap
from flask import *
import matplotlib.pyplot as plt
from func import *


app = Flask(__name__)
data = load_csv()
variables = {}
variable_names = ["Variable1","Variable2","Variable3","Variable4","Variable5"]


@app.route('/costOL_heatmap')
def costOL_heatmap():
    return render_template("costOL_heatmap.html")



@app.route('/')
def index():
    return render_template("index.html")





@app.route('/processing')
def processing():
    try:
        variable1 = request.args.get('variable1')
        variable2 = request.args.get('variable2')
        variable3 = request.args.get('variable3')
        variable4 = request.args.get('variable4')
        variable5 = request.args.get('variable5')
        Search_town = request.args.get('Dublin')

        variables["Variable1"] = variable1
        variables["Variable2"] = variable2
        variables["Variable3"] = variable3
        variables["Variable4"] = variable4
        variables["Variable5"] = variable5
    
        
        if Search_town in data['town'].values: # if the town is in the dataset AC
            row = data[data['town'] == Search_town].iloc[0] # Find the row corresponding to the town AC
            # extracts longitude and latitude AC
            Long1 =  (row['longitude'])
            Lat1 = (row['latitude'])
        
        factors, costOL = generate_realD_from_csv(data,variables)


        latentDim = 10  # latent dimension for generator AC
        epochs = 10 # epochs AC
        batchS = 128 # batch size AC

        # build models AC
        generator = build_generator(latentDim)
        discriminator = build_discriminator()

        discriminator.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy']) # compiles the discriminator model AC

        gan = build_gan(generator, discriminator,latentDim) # builds GAN model and compiles it AC


        realD = generate_realD_from_csv(data,variables)

        # train the GAN AC
        train_gan(generator, discriminator, gan, data, epochs, batchS, latentDim,variables)
        calculate_factor_importance(generator, latentDim) # calculates factor importance AC
        numSamp = 1000
        generated_costOL = generator.predict(np.random.rand(numSamp, latentDim))
        factorImportance = calculate_factor_importance_and_plot(generator, latentDim, data)
        # generates synthetic data AC
        heatyMax=(max(factorImportance))#declares highest importance
        factorNames = ['Housing', 'Transport', 'Energy', 'Food', 'Education']
        
        heattMaxPos=factorImportance.index(heatyMax)
        get_pos=factorNames[heattMaxPos]
        
        
        
        hMap = folium.Map(location=[Lat1, Long1], zoom_start=12)

        # adds a marker to the map AC
        folium.Marker([Lat1, Long1], popup="Your Location").add_to(hMap)

        # explicitly set heatmap data AC
        heatData = [[row['latitude'], row['longitude'], row['cost_of_living']] for index, row in data.iterrows()]
        # attempt for heatnap colouring to match factors below in quotes AC
        '''color_scheme=get_pos
        if color_scheme =='Housing':
            gradient = {0.2: 'red'}
        elif color_scheme == 'Transport':
            gradient = {0.2: 'blue'}
        elif color_scheme == 'Energy':
            gradient = {0.2: 'green'}
        elif color_scheme == 'Food':
            gradient = {0.2: 'yellow'}
        else:
            gradient = {0.2: 'purple'}'''
        # adds heat data to map AC
        HeatMap(heatData).add_to(hMap)

        # save the heatmap to an HTML file AC
        hMap.save('templates/costOL_heatmap.html')
        fig, axs = plt.subplots(2, 1, figsize=(10, 12))

        # First plot (bar graph for factor importance) AC
        axs[0].bar(['Housing', 'Transport', 'Energy', 'Food', 'Education'], factorImportance, color=['red','blue','green','yellow','purple'])
        axs[0].set_xlabel('Factors')
        axs[0].set_ylabel('Importance')
        axs[0].set_title('Importance of Each Factor on Cost of Living')

        # Second plot (scatter plot for generated data) AC
        numSamp = 500
        scale_factor=50
        noise_level=15
        sum_of_factors=np.sum(factors,axis=1)
        sum_of_factors = np.sum(np.random.rand(numSamp, 5), axis=1)
        generated_costOL=sum_of_factors*scale_factor+np.random.normal(0,noise_level,numSamp)
        axs[1].scatter(sum_of_factors, generated_costOL, label='Generated Data')
        axs[1].set_xlabel('Sum of Factors')
        axs[1].set_ylabel('Cost of Living')
        axs[1].legend()

        
        fig.savefig('static/factor_importance.png', bbox_inches='tight')
        return jsonify({"status":True,"messages":"HeatMap created"}) 
    except:
        return jsonify({"status":False,"messages":"HeatMap Not created"})

if __name__ == '__main__':
    app.run(host='127.0.0.1',debug=True, port=5004) 
