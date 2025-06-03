# Traffic Collision Prediction with Graph Attention Networks (GAT)

## Overview

This project implements **Graph Attention Networks (GAT)** for traffic accident prediction. In this model:
- Vehicles are represented as individual **nodes** in a graph.
- Vehicle classes serve as key **node features**.

The primary goal of this prediction framework is **graph classification**, specifically to predict the occurrence of a traffic accident.

## Key Achievements

- The GNN model achieves **86.1% accuracy** in graph classification tasks.
- This framework offers promising avenues for enhancing collision warning systems and improving overall road safety measures.

## Project Structure & Code

The core implementations and experimental results can be found in the following Jupyter notebooks:

- **`Single_Graph_BC.ipynb`**: Contains the code and experimental results for the GAT + Binary Classification model.
- **`LSTM_Node_GAT.ipynb`**: Includes the code and experimental results for the LSTM + GAT + Binary Classification model.

## Dataset

The data used for this project is organized within the `csv_final` directory:

- **`csv_final/normal/`**: Contains data for normal (non-accident) driving scenes.
- **`csv_final/accident/`**: Contains data for accident scenes.
- **`csv_final/info/`**: Provides detailed information for each of the accident scenes.

## Future Work
-   Explore real-time prediction capabilities.
-   Incorporate more diverse environmental data.

## Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue.

## License
This project is licensed under the MIT License - see the `LICENSE.md` file for details.
