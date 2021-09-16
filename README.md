# Análisis cuantitativo de la marcha 

 En este proyecto se ha combinado la captura y análisis de datos sensoriales con la aplicación de técnicas de *Machine Learning*, con el objetivo de construir una herramienta que permite automatizar, agilizar y objetivar la valoración de la marcha humana, y en particular, el empleo de una de las escalas funcionales más extendidas en el contexto de la práctica clínica: la **escala de Tinetti**.

La siguiente figura resume, de manera esquemática, el trabajo realizado. 

![Esquema general](docs/images/esquema-general.svg)

Se ha construido un *dataset* que sirve de línea base para la valoración automática de la marcha, utilizando como fuente de adquisición de datos un sistema de *tracking* óptico profesional para la captura y análisis de movimiento, basado en cámaras y marcadores reflectantes (**OptiTrack + Clinical 3DMA**). Para exportar los datos desde Clinical 3DMA se ha implementado un *script* basado en la automatización de la GUI, cuyo código fuente se incluye en el directorio ***3dma_exporter***.

Este conjunto de datos es accesible desde *Kaggle*, una comunidad gratuita con recursos orientados a la ciencia de datos y el *Machine Learning*:

www.kaggle.com/serglez/gait-dataset-using-a-3d-motion-capture-system


Después de la recopilación de datos inicial, se ha llevado a cabo una serie de actividades para preparar el conjunto de datos, incluyendo tareas de limpieza de datos y procesos de ingeniería de características para crear variables explicativas adicionales, tales como parámetros espaciotemporales o curvas cinemáticas de la marcha a partir de los datos en crudo procedentes del sistema de sensorización. El *script* implementado para esta tarea encuentra en el directorio ***preprocessor-cli***.

Sobre los datos preprocesados, se han entrenado y evaluado clasificadores binarios basados en diversos algoritmos de Machine Learning, con la intención de obtener varios modelos predictivos que permitan generar automáticamente las respuestas a las distintas categorías que constituyen el criterio de marcha en la mencionada prueba de Tinetti. El código asociado se encuentra en el directorio ***notebooks***. 

Finalmente, los modelos seleccionados (incluidos en formato *.pkl* en la carpeta ***models***) han sido desplegados e integrados en una herramienta que genera informes detallados con análisis cuantitativos de la marcha humana (***report-generator***).
