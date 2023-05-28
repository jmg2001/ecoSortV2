import streamlit as st

def inicio():
    import streamlit as st

    st.title(":green[EcoSort] ♻️",
            )

    st.subheader("¡Bienvenido a nuestra plataforma de clasificación de residuos con inteligencia artificial!")

    st.markdown(
        """
        Esto es :green[EcoSort], tu asistente para clasificar tu basura de manera adecuada. En nuestro sitio web, hemos creado una 
        solución innovadora que utiliza algoritmos de inteligencia artificial para ayudarte a clasificar diferentes tipos 
        de residuos de manera eficiente y precisa. Con solo unos pocos clics, podrás identificar y categorizar residuos comunes, 
        contribuyendo así a un manejo adecuado y sostenible de los desechos.

        Nuestra plataforma combina la potencia del aprendizaje automático y la visión por computadora para analizar imágenes de 
        residuos. Simplemente carga una foto del residuo que deseas clasificar, y nuestra inteligencia artificial se encargará de 
        procesarla rápidamente, proporcionándote una clasificación precisa en cuestión de segundos.
        
        ### ¿Cómo funciona?
        
        Nuestra plataforma utiliza tecnología de inteligencia artificial para realizar la clasificación precisa de diferentes 
        tipos de residuos. El funcionamiento se basa en algoritmos de aprendizaje automático que han sido entrenados con una 
        amplia variedad de imágenes de residuos.

        Cuando cargas una imagen de residuo en nuestra plataforma, nuestro sistema de inteligencia artificial analiza automáticamente 
        las características visuales del objeto y las compara con un extenso conjunto de datos de entrenamiento. Esto nos permite 
        reconocer patrones y características distintivas que nos ayudan a determinar el tipo de residuo con una alta precisión.

    """
    )


def puntos_recoleccion():
    import streamlit as st
    import pandas as pd
    from streamlit_folium import st_folium
    import folium
    import statistics
    from folium.plugins import MarkerCluster

    @st.cache_data
    def get_data(path):
        data = pd.read_csv(path)
        return data

    data = get_data("puntosRecoleccion.csv")

    nombres = data.iloc[:,5]
    print(nombres[0])
    locacion = data.iloc[:,10]
    horario_fin = data.iloc[:,9]
    horario_inicio = data.iloc[:,8]
    dias = data.iloc[:,7]
    locacionL = list()
    latitudes = list()
    longitudes = list()
    for dato in locacion:
        latitudes.append(float(dato.split(",")[0]))
        longitudes.append(float(dato.split(",")[1]))
        locacionL.append([float(dato.split(",")[0]),float(dato.split(",")[1])])

    m = folium.Map(location=[statistics.mean(latitudes),statistics.mean(longitudes)], zoom_start=10)
    marker_cluster = MarkerCluster().add_to(m)

    #@st.cache_data
    def create_marks():
        for i in range(0,len(locacion)):
            texto = f"""Locacion: {nombres[i]}
                        Horario Inicio: {horario_inicio[i]}
                        Horario Fin: {horario_fin[i]}
                        Dias: {dias[i]}"""
            folium.Marker(
                locacionL[i],
                popup=texto,
                tooltip=nombres[i],
            ).add_to(marker_cluster)

    create_marks()

    st.title("Puntos de recoleción 🚛")

    st.markdown("""
            Aquí podrás identifar el punto de recollecion de residuos más cercano a tu domicilio
                """)

    st_data = st_folium(m, width=725)

    st.markdown("""
            Esta herramienta te permite visualizar de manera precisa los **diferentes puntos de recolección**
            de basura dentro del área metropolitana, conforme te acercas a una zona en especial se puede observar
            con mayor detalle la ubicación de los puntos.

            Si se desea conocer **información más específica** como el horario y días de recolección,se puede hacer
            click sobre cualquier punto y dicha información se desplegará.
                """)
    st.button("Re-run")

def tira_basura():
    import streamlit as st
    import cv2
    import numpy as np
    import tensorflow as tf
    from tensorflow import keras
    import tensorflow_hub as hub
    from streamlit_extras.let_it_rain import rain


    class_names = ['Cartón', 'Vidrio', 'Metal', 'Papel', 'Plastico', 'Basura'] #For model 85
    #class_names = ['Bateria', 'Biodegradable', 'Vidrio Cafe', 'Carton', 'Ropa', 'Vidrio Verde', 'Aluminio','Papel','Plastico','Zapatos','Basura','Vidrio Blanco']  #For model 93

    @st.cache_resource
    def create_model(path):
        model = keras.models.load_model(path,custom_objects={'KerasLayer':hub.KerasLayer})
        return model
    count = 0
    if count == 0:
        modelo = create_model("modelo_85.h5")
        count = 1


    st.title("Tira tu basura aquí:",)

    st.markdown(
        """
        Abre la camara y toma una foto a tu basura, nuestro algoritmo de clasificación
        de imagenes utilizando ``Tensorflow`` nos permite poder decidir que tipo de basura
        tienes.

        """
    )

    img_file_buffer = st.camera_input("Toma una foto",label_visibility="hidden")

    if img_file_buffer is not None:
        # To read image file buffer with OpenCV:
        bytes_data = img_file_buffer.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        img_res = cv2.resize(cv2_img, (224,224))
        img_res = img_res/255.0
        imagenFinal = tf.expand_dims(img_res, axis=0)

        prediccion = modelo.predict(imagenFinal)
        prediccion = tf.squeeze(prediccion)

        if max(prediccion) > 0.70:

            maxIndex = tf.argmax(prediccion).numpy()

            btn = st.download_button(
                    label="Guardar",
                    data=bytes_data,
                    file_name="Basura.png",
                    mime="image/png"
                )
            
            st.title(f"Tu residuo fue clasificado como: :green[{class_names[maxIndex]}]")

            text = str()

            if maxIndex == 0:
                emoji = "📦"
                text = "El cartón, utilizado en el embalaje y en la fabricación de cajas y envases, puede tener impactos ambientales negativos. Su producción requiere la tala de árboles, lo que puede llevar a la deforestación y la pérdida de hábitats naturales. Además, el proceso de fabricación del cartón consume grandes cantidades de agua y energía, contribuyendo a la escasez de recursos y a la emisión de gases de efecto invernadero. Cuando el cartón se desecha incorrectamente, ya sea en vertederos o incineradoras, puede contribuir a la contaminación del suelo, del agua y del aire. Sin embargo, el cartón es un material altamente reciclable y su reciclaje adecuado puede reducir significativamente su impacto ambiental."
            if maxIndex == 1:
                emoji = "🍾"
                text = "A diferencia del plástico, el vidrio es un material que no genera contaminantes durante su producción. Sin embargo, su fabricación requiere grandes cantidades de energía y emite dióxido de carbono (CO2). Cuando el vidrio se desecha incorrectamente, puede convertirse en un problema ambiental. Los desechos de vidrio pueden tardar miles de años en descomponerse en la naturaleza y ocupan espacio en los vertederos. Si se queman en incineradoras, pueden liberar gases contaminantes al aire. Además, el vidrio roto puede representar un peligro para la vida silvestre y los seres humanos si no se maneja adecuadamente. Sin embargo, el vidrio es altamente reciclable y su reciclaje reduce significativamente su impacto ambiental."
            if maxIndex == 2:
                text = "El metal puede tener diversos impactos ambientales negativos a lo largo de su ciclo de vida. La extracción de minerales metálicos implica la destrucción de ecosistemas, deforestación y contaminación del suelo y el agua. El proceso de refinado y fabricación de metales requiere grandes cantidades de energía y emite gases de efecto invernadero, contribuyendo al cambio climático. Además, muchos metales contienen sustancias tóxicas, como el plomo o el mercurio, que pueden contaminar el suelo, el agua y la vida silvestre. La disposición inadecuada de los desechos de metal también puede generar contaminación del suelo y del agua. Sin embargo, el reciclaje del metal es fundamental para reducir la necesidad de extracción y minimizar su impacto ambiental, conservando recursos naturales y evitando la contaminación asociada a su producción."
                emoji = "⚙️"
            if maxIndex == 3:
                text = "La producción y el uso del papel pueden tener impactos ambientales negativos. La fabricación de papel requiere la tala de árboles, lo que puede llevar a la deforestación y la pérdida de biodiversidad. Además, el proceso de fabricación consume grandes cantidades de agua, energía y productos químicos, contribuyendo a la contaminación del agua y la emisión de gases de efecto invernadero. El desecho de papel inadecuado también es problemático, ya que la mayoría de los residuos de papel terminan en vertederos, donde se descomponen y liberan metano, un gas de efecto invernadero potente. Sin embargo, el papel es altamente reciclable y su reciclaje adecuado puede reducir significativamente su impacto ambiental, conservando recursos naturales y reduciendo la contaminación."
                emoji = "📄"
            if maxIndex == 4:
                text = "El plástico contamina el medio ambiente de varias formas. Su producción a partir de combustibles fósiles emite gases de efecto invernadero, contribuyendo al cambio climático. Además, el plástico es duradero y puede tardar cientos de años en descomponerse, lo que lleva a la acumulación de desechos en vertederos y océanos. Cuando el plástico se descompone en microplásticos, puede ser ingerido por animales marinos, causando daños en su salud y en toda la cadena alimentaria. La quema de plástico libera sustancias tóxicas en el aire. La gestión inadecuada de los residuos plásticos también puede causar contaminación del suelo y del agua, impactando negativamente en los ecosistemas y la vida humana."
                emoji = "🔫"
            if maxIndex == 5:
                text = "La basura en general tiene diversos impactos ambientales negativos. Cuando la basura se acumula en vertederos, produce la liberación de gases de efecto invernadero, como metano, que contribuyen al cambio climático. La gestión inadecuada de la basura puede contaminar el suelo y el agua, afectando los ecosistemas naturales y la vida marina. Además, la incineración de residuos produce emisiones tóxicas y contribuye a la contaminación del aire. La basura también representa una gran demanda de recursos naturales, como energía y agua, para su producción y eliminación. Para mitigar estos impactos, es crucial promover la reducción, reutilización, reciclaje y una gestión adecuada de la basura."
                emoji = "🗑️"
            
            # if maxIndex == 0:
            #     text = "Las baterías contienen metales tóxicos como el plomo, el cadmio, el mercurio y el litio, entre otros. Cuando las baterías se desechan incorrectamente, estos metales pueden filtrarse en el suelo y las aguas subterráneas, contaminando el medio ambiente. Además, si las baterías se incineran o rompen, los metales tóxicos pueden liberarse al aire, causando contaminación atmosférica. Estos metales pueden acumularse en los organismos vivos a lo largo de la cadena alimentaria, causando daños en la salud humana y en los ecosistemas. Por lo tanto, es esencial reciclar adecuadamente las baterías para minimizar su impacto negativo en el medio ambiente."
            #     emoji = "🔋"
            # if maxIndex == 1:
            #     text = "Los residuos biodegradables, como los restos de alimentos y materiales orgánicos, pueden tener varios impactos en el ambiente. Por un lado, cuando se manejan adecuadamente, pueden ser una fuente de nutrientes para el suelo, favoreciendo la fertilidad y el crecimiento de las plantas. Además, su procesamiento mediante compostaje o digestión anaeróbica puede generar biogás, una forma de energía renovable. Sin embargo, si estos residuos no se gestionan adecuadamente y terminan en vertederos, pueden generar emisiones de gases de efecto invernadero, como metano, contribuyendo al cambio climático. También pueden causar contaminación del suelo y del agua si se descomponen sin control. Por lo tanto, es fundamental implementar prácticas adecuadas de gestión de residuos biodegradables, como el compostaje y la recolección selectiva, para maximizar sus beneficios y minimizar sus impactos negativos en el ambiente."
            #     emoji = "♻️"
            # if maxIndex == 2 or maxIndex == 5 or maxIndex == 11:
            #     text = "A diferencia del plástico, el vidrio es un material que no genera contaminantes durante su producción. Sin embargo, su fabricación requiere grandes cantidades de energía y emite dióxido de carbono (CO2). Cuando el vidrio se desecha incorrectamente, puede convertirse en un problema ambiental. Los desechos de vidrio pueden tardar miles de años en descomponerse en la naturaleza y ocupan espacio en los vertederos. Si se queman en incineradoras, pueden liberar gases contaminantes al aire. Además, el vidrio roto puede representar un peligro para la vida silvestre y los seres humanos si no se maneja adecuadamente. Sin embargo, el vidrio es altamente reciclable y su reciclaje reduce significativamente su impacto ambiental."
            #     emoji = "🍾"
            # if maxIndex == 3:
            #     emoji = "📦"
            #     text = "El cartón, utilizado en el embalaje y en la fabricación de cajas y envases, puede tener impactos ambientales negativos. Su producción requiere la tala de árboles, lo que puede llevar a la deforestación y la pérdida de hábitats naturales. Además, el proceso de fabricación del cartón consume grandes cantidades de agua y energía, contribuyendo a la escasez de recursos y a la emisión de gases de efecto invernadero. Cuando el cartón se desecha incorrectamente, ya sea en vertederos o incineradoras, puede contribuir a la contaminación del suelo, del agua y del aire. Sin embargo, el cartón es un material altamente reciclable y su reciclaje adecuado puede reducir significativamente su impacto ambiental."
            # if maxIndex == 4:
            #     emoji = "👕"
            #     text = "La industria de la moda y la ropa puede tener varios impactos ambientales negativos. El proceso de producción de prendas implica el uso intensivo de agua, energía y productos químicos, lo que contribuye a la contaminación del agua, la emisión de gases de efecto invernadero y la degradación del suelo. Además, el uso de fibras sintéticas como el poliéster libera microplásticos en el agua durante el lavado, lo que afecta a los ecosistemas acuáticos. El desecho de ropa también representa un problema, ya que la mayoría de las prendas terminan en vertederos, donde pueden tardar años en descomponerse. La moda rápida y el consumo excesivo de ropa también fomentan un ciclo de desperdicio y agotamiento de recursos naturales. Para mitigar estos impactos, es importante optar por materiales sostenibles, reciclar y donar la ropa en lugar de desecharla, y promover una mentalidad de consumo responsable."
            # if maxIndex == 6:
            #     text = "La producción y el uso del aluminio pueden tener impactos ambientales negativos significativos. La extracción del mineral de aluminio requiere grandes cantidades de energía y puede resultar en la degradación del suelo y la deforestación. Además, el proceso de producción del aluminio a partir de la bauxita emite gases de efecto invernadero, como dióxido de carbono y perfluorocarbonos, contribuyendo al cambio climático. La producción de aluminio también genera residuos tóxicos, como lodos rojos, que pueden contaminar el agua y el suelo. Sin embargo, el aluminio es altamente reciclable, lo que reduce la necesidad de extracción y reduce el impacto ambiental. El reciclaje del aluminio requiere menos energía y emite menos gases de efecto invernadero en comparación con la producción primaria, lo que lo convierte en una opción más sostenible."
            #     emoji = "🥫"
            # if maxIndex == 7:
            #     text = "La producción y el uso del papel pueden tener impactos ambientales negativos. La fabricación de papel requiere la tala de árboles, lo que puede llevar a la deforestación y la pérdida de biodiversidad. Además, el proceso de fabricación consume grandes cantidades de agua, energía y productos químicos, contribuyendo a la contaminación del agua y la emisión de gases de efecto invernadero. El desecho de papel inadecuado también es problemático, ya que la mayoría de los residuos de papel terminan en vertederos, donde se descomponen y liberan metano, un gas de efecto invernadero potente. Sin embargo, el papel es altamente reciclable y su reciclaje adecuado puede reducir significativamente su impacto ambiental, conservando recursos naturales y reduciendo la contaminación."
            #     emoji = "📄"
            # if maxIndex == 8:
            #     text = "El plástico contamina el medio ambiente de varias formas. Su producción a partir de combustibles fósiles emite gases de efecto invernadero, contribuyendo al cambio climático. Además, el plástico es duradero y puede tardar cientos de años en descomponerse, lo que lleva a la acumulación de desechos en vertederos y océanos. Cuando el plástico se descompone en microplásticos, puede ser ingerido por animales marinos, causando daños en su salud y en toda la cadena alimentaria. La quema de plástico libera sustancias tóxicas en el aire. La gestión inadecuada de los residuos plásticos también puede causar contaminación del suelo y del agua, impactando negativamente en los ecosistemas y la vida humana."
            #     emoji = "🔫"
            # if maxIndex == 9:
            #     text = "Los zapatos pueden tener varios impactos ambientales negativos a lo largo de su ciclo de vida. Durante su fabricación, se utilizan materiales que requieren energía y recursos naturales, como cuero, caucho y textiles, lo que puede generar emisiones de gases de efecto invernadero y contaminación del agua. Además, los procesos de teñido y tratamiento químico de los materiales pueden liberar sustancias tóxicas al medio ambiente. Al final de su vida útil, muchos zapatos terminan en vertederos, donde se descomponen lentamente y liberan gases de efecto invernadero. Para mitigar estos impactos, es importante optar por zapatos fabricados de manera sostenible, utilizar materiales reciclados y reciclables, y considerar la reparación o donación en lugar de desecharlos."
            #     emoji = "👞"
            # if maxIndex == 10:
            #     text = "La basura en general tiene diversos impactos ambientales negativos. Cuando la basura se acumula en vertederos, produce la liberación de gases de efecto invernadero, como metano, que contribuyen al cambio climático. La gestión inadecuada de la basura puede contaminar el suelo y el agua, afectando los ecosistemas naturales y la vida marina. Además, la incineración de residuos produce emisiones tóxicas y contribuye a la contaminación del aire. La basura también representa una gran demanda de recursos naturales, como energía y agua, para su producción y eliminación. Para mitigar estos impactos, es crucial promover la reducción, reutilización, reciclaje y una gestión adecuada de la basura."
            #     emoji = "🗑️"

            st.markdown(text)

            rain(
                emoji=emoji,
                font_size=54,
                falling_speed=1,
                animation_length=1,
            )
        else:
            st.title(f"Lo siento:( no pudimos clasificar tu :green[residuo]")

def beneficios():
    import streamlit as st
    import pandas as pd
    from streamlit_folium import st_folium
    import folium
    import statistics
    from folium.plugins import MarkerCluster

    st.title("Características 🪄:")
    st.markdown("""
    - **Precisión y confiabilidad**: El proyecto utiliza algoritmos de inteligencia artificial y técnicas avanzadas de procesamiento de imágenes para lograr una clasificación precisa de los diferentes tipos de residuos. Esto garantiza resultados confiables y evita errores comunes en la clasificación manual.

    - **Eficiencia y velocidad**: La clasificación de residuos mediante inteligencia artificial es un proceso rápido y eficiente. Los resultados se obtienen en cuestión de segundos, lo que permite ahorrar tiempo y recursos en comparación con los métodos de clasificación tradicionales.

    - **Amplia variedad de tipos de residuos**: El sistema está diseñado para reconocer y clasificar una amplia gama de tipos de residuos, incluyendo papel, plástico, vidrio, metal, cartón, materia orgánica, electrónicos, entre otros. Esto asegura que puedas clasificar diferentes tipos de residuos con una sola plataforma.

    - **Facilidad de uso**: El proyecto se desarrolla con una interfaz de usuario intuitiva y fácil de usar. Los usuarios pueden capturar imágenes de los residuos a través de dispositivos móviles o cargar imágenes desde sus computadoras, lo que simplifica el proceso de clasificación.
    """)

    st.title("Beneficios 🌱:")
    st.markdown("""
    - **Promoción de la conciencia ambiental**: Al facilitar la clasificación adecuada de los residuos, el proyecto contribuye a aumentar la conciencia ambiental y promover prácticas de reciclaje adecuadas. Esto ayuda a reducir la contaminación y el impacto negativo en el medio ambiente.

    - **Optimización de la gestión de residuos**: La clasificación precisa de los residuos proporciona información valiosa para mejorar la gestión de residuos. Permite identificar patrones y tendencias en la generación de residuos, lo que puede ayudar a tomar decisiones informadas sobre estrategias de reciclaje, reutilización y reducción de residuos.

    - **Facilita el reciclaje y la reutilización**: Al identificar correctamente los tipos de residuos, el proyecto facilita el proceso de reciclaje y reutilización. Los usuarios pueden tomar decisiones más informadas sobre cómo desechar los residuos, identificar los puntos de reciclaje adecuados y fomentar la separación adecuada de los materiales reciclables.

    - **Contribuye a un futuro sostenible**: Al utilizar tecnologías avanzadas como la inteligencia artificial, el proyecto se alinea con los objetivos de sostenibilidad y protección del medio ambiente. Ayuda a construir un futuro más limpio y sostenible al fomentar prácticas responsables de manejo de residuos y promover la economía circular.
    """)

page_names_to_funcs = {
    "Inicio ♻️": inicio,
    "Tira tu basura 🗑️": tira_basura,
    "Puntos de recoleccion 🚛": puntos_recoleccion,
    "Características y beneficios 🌱": beneficios
}

demo_name = st.sidebar.selectbox("¿Qué quieres hacer?:", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()