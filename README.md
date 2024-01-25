Instituto de Investigaciones en Inteligencia Artificial - Universidad Veracruzana

**Aprendizaje y clasificación de documentos**

- Reunir un conjunto de 15 documentos de 5 clases. Cada documento de al menos 1 pagina. Seleccionar 10 documentos de cada clase aleatoriamente para realizar el entrenamiento bajo el esquema de bolsa de palabras. 
- Emplear los 5 documentos restantes de cada clase para clasificar empleando el teorema de Bayes.
- Resumir los resultados empleando una matriz de confusión.
- Calcular métricas TP, TN, FP, FN por clase, Accuracy, Precisión, Recall, F1-score por clase.
- Discutir los resultados obtenidos.

Para la realización de este trabajo se utiliza python como lenguaje de programación, ademas de hacer uso de librerías como numpy, pandas y scikitlearn, esta ultima debido a que tiene funciones especificas que simplifican el desarrollo del programa, una de ellas es *CountVectorizer* que sirvió para crear la bolsa de palabras con los textos de entrenamiento, y crear vectores para cada texto con las diferentes palabras que tienen, otra es *MultinomialNB* la cual se utilizo para entrenar el modelo con la bolsa de valores generada, y posteriormente evaluar los nuevos textos de test, por ultimo se utilizo la función *metrics* para obtener estadísticas como la matriz de confusión y el Accuracy del modelo.

Los textos utilizados son quince libros en cinco categorías diferentes, las cuales son: Artificial Intelligence, Business, Education, Fantasy y Health; todos los libros están en idioma ingles y se convirtieron a formato .txt, ademas de que todos los textos están en minúsculas para evitar la posibilidad de que se tomaran como diferentes palabras si una estaba en mayúsculas y en minúsculas en diferentes partes de los textos. Ademas en el código para realizar la clasificación se les coloco un numero a cada clase: Artificial Intelligence=0, Business=1, Education=2, Fantasy=3 y Health=4.



Los resultados obtenidos son los siguientes: La matriz de confusión es:

[[4 0 0 0 1]

[0 4 0 1 0]

[0 0 5 0 0]

[1 0 0 4 0]

[0 1 0 0 4]]


El Accuracy es: 0.84


Recalls:

R0=0.8 R1=0.8 R2=1.0 R3=0.8 R4=0.8


El Macro Average Recall es: 0.8400000000000001


Precisiones: P0=0.8 P1=0.8 P2=1.0 P3=0.8 P4=0.8


El Macro Average Precisión es: 0.8400000000000001


F1-scores: F0=0.8000000000000002 F1=0.8000000000000002 F2=1.0 F3=0.8000000000000002 F4=0.8000000000000002


Matrices de confusión de cada clase:

MC0=

[[ 4  1]

[ 1 19]]

MC1=

[[ 4  1]

[ 1 19]]

MC2=

[[ 5  0]

[ 0 20]]

MC3=

[[ 4  1]

[ 1 19]]

MC4=

[[ 4  1]

[ 1 19]]


Como se puede ver en los resultados el Accuracy obtenido es de 84%, ya que la mayoría de las clases clasificaron erróneamente un libro, excepto por la clase 2, Education, que clasifico de manera correcta sus cinco libros de test, las demás clases aunque si clasificaron en total cinco libros para cada una, uno de esos libros pertenecía a una clase diferente, es por eso que sus Recall, precisión y F1-score son de 0.8, esto es mas notorio en las matrices de confusión de cada una de las clases donde se aprecia que tienen un falso positivo y un falso negativo respectivamente, pero en general se obtiene un porcentaje aceptable de clasificación, aunque posiblemente podría ser mejorado si se le agregan mas ejemplos de entrenamiento. Por otra parte gracias a la librería de scikitlearn el programa se ejecuta de manera rápida, tardando menos de un minuto en ejecutarse y obtener los resultados esperados, por lo que es una herramienta muy útil, que al estar tan bien optimizada para python es posible realizar clasificaciones muy rápidas.

Métodos Probabilísticos para la Inteligencia Artificial
