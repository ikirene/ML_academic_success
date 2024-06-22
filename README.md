![Machine Learning Academic Success](https://inoxoft.com/wp-content/uploads/2021/02/Image-1-80-1.jpg)

# MACHINE LEARNING PROJECT (Kaggle Competition)
## **[CLASSIFICATION WITH AN ACADEMIC SUCCESS DATASET](https://www.kaggle.com/competitions/playground-series-s4e6/overview)** 🎓📖
Este proyecto forma parte de una de las competiciones de Kaggle, la cual se basa en el entrenamiento de modelos de Machine Learning para la predicción del riesgo académico de los estudiantes en educación superior. Para abordar este problema, vamos a trabajar con 3 arvchivos (.csv). El primero lo utilizaremos para entrenar nuestros modelos, el segundo para testearlos y el tercero nos servirá como herramienta para competir. 

## **DESCRIPCIÓN DATASET**

El conjunto de datos para esta competencia (tanto de entrenamiento como de prueba) fue generado a partir de un modelo de aprendizaje profundo 
entrenado en el conjunto de datos "Predict Students' Dropout and Academic Success". Las distribuciones de características son similares, 
pero no exactamente iguales, a las del conjunto de datos original. Siéntete libre de utilizar el conjunto de datos original como parte de esta competencia, 
tanto para explorar las diferencias como para ver si incorporar el original en el entrenamiento mejora el rendimiento del modelo. 
Por favor, consulta el conjunto de datos original para obtener explicaciones de las características.

## **OBJETIVOS DEL PROYECTO**

1. Extraer los datos de forma organizada y óptima.
2. Hacer una lectura de la descripción de las variables obtenidas. 
3. Preprocesamiento de los datos. Aplicamos herramientas y técnicas para preparar los datos según nuestras necesidades:
   - Observación y limpieza de valores nulos, si procede.
   - Observación y clasificación del tipo de variables (continuas o categóricas).
   - Filtrado de los datos extraidos, selección de variables y observación de las correlaciones.
   - Transformación de diferentes conjuntos de datos.
   - Creación de funciones que agilicen el preprocesamiento y la implementación de los modelos.
   - Librerías utilizadas: Pandas, Numpy, Seaborn, Sklearn.
   
5. Aplicar modelos de Machine Learning para problemas de clasificación (DecisionTree, RandomForest, LogisticRegression o KNNeighbors).
6. Evaluar dichos modelos y elegir es más estable.
7. Hacer predicciones con nuestros modelos en base al dataset provisto para el test.
8. Exportar y cargar en Kaggle nuestras predicciones.

## **PREPROCESAMIENTO DE LOS DATOS**

Me encuentro con un _dataset_ con gran cantidad de registros y variables (76518 rows × 38 columns). En términos de variables, podemos hacer las siguientes observaciones:

- En la variable objetivo ('Target') puedo observar que hay 3 categorías: Graduate, Dropout y Enrolled. Esto significa que no solo es un problema de clasificación, sino que además es multicategórico. Debo tener esto en cuenta a la hora de entrenar y testear mis modelos.
  
- Analizo la relación entre variables y descartamos aquellas que no aportan demasiado. Por ejemplo, algunas variables aportan información irrelevante o redundante, como 'id' o 'International'. Si vemos los porcentajes de 'Nationality', el 99% de los estudiantes son de Portugal (locales), por lo que no tiene sentido conservar la columna 'Interational' que nos indica si el estudiante es local o internacional.
  
- Aplico la técnica OneHotEncoder para obtener los dummies de nuestras variables categóricas. Esta técnica es mucho más viable que la función .get_dummies, ya que evita posibles problemas en caso de encontrar una categoría nueva en el testeo.

## **ENTRENAMIENTO Y TESTEO DE MODELOS**

Una vez tenemos los datos preparados, voy a lanzar dos modelos sencillos y observar el resultado, LogisticRegressor y DecisionTree. Por la naturaleza del algoritmo de DT y su tendencia al _overfitting_ , intuyo que obtendré un mejor resultado con LogisticRegressor con un problema de multicagorías como este. A partir de los primeros resultados con cada modelo, la labor consistirá en hacer pequeñas modificaciones en los datos y su limpieza para intentar exprimir nuestro modelo al máximo.

## **CONCLUSIONES**

Al realizar este proyecto me di cuenta de la infinidad de modelos que existen para realizar nuestras predicciones. Por el momento en el que hice este proyecto no conocía las herramientas como PyCaret, H20 o Lazy, que me hubieran ayudado a elegir el modelo idóneo. Sin embargo, veo que lo que marca la diferencia entre una predicción u otra es el proceso de limpieza, preparación y transformación de los datos. Si no somos cuidadosos con este proceso, nuestros modelos carecerán de la consistencia y fiabilidad necesaria para ser tomados como válidos.












