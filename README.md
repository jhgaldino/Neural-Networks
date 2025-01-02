## Neural-Networks: Previsão de Resultados de Futebol

### Introdução

Este projeto utiliza redes neurais para prever os resultados da última rodada de um campeonato de futebol. Com foco em técnicas de machine learning, o projeto visa oferecer insights sobre como diferentes parâmetros da rede neural podem influenciar nas previsões.

### Parâmetros do Modelo

Para explorar diferentes resultados, os seguintes parâmetros podem ser ajustados:

hidden_layers (int > 0): Define o número de camadas ocultas na rede, influenciando a complexidade do modelo.

epochs (int > 0): Quantidade de ciclos de treinamento do modelo.

lr (float entre 0 e 1): Learning rate, determina a velocidade de aprendizagem do modelo.

momentum (float entre 0 e 1): Contribui para a atualização dos pesos durante o treinamento, afetando a convergência do modelo.

### Preparação dos Dados

Os dados são inicialmente limpos, removendo colunas desnecessárias e normalizando valores. O conjunto de dados é então dividido em partes de treinamento e teste.

### Estrutura do Modelo

O modelo utiliza uma arquitetura de rede neural especificada pelo usuário, incluindo número de camadas e neurônios. A rede é treinada com os dados de entrada e ajustada com base nos parâmetros definidos.

### Executando o Modelo

Este projeto é um Jupyter Notebook. Execute cada célula do notebook para ver o processo passo a passo. Isso inclui a carga dos dados, a preparação, a definição do modelo, o treinamento e a visualização dos resultados.

### Contribuições
Sugestões e contribuições para o projeto são bem-vindas. Para dúvidas ou colaborações, por favor, entre em contato.
