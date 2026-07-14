# Image Encryption using GTSMn and Henon Map

A Python-based digital image encryption application that combines **GTSMn (Generating Tree of Symmetric Group / Production Matrix)**, **Shuffle Permutation**, and **Henon Map Chaotic Diffusion** to provide secure image encryption and decryption with comprehensive security analysis.

---

## Features

- Image Encryption
- Image Decryption
- Password-Based Key Generation
- GTSMn Matrix Encryption
- Shuffle Permutation
- Henon Map Chaotic Diffusion
- Histogram Analysis
- Security Analysis
- Adjacent Pixel Correlation Analysis
- NPCR & UACI Evaluation
- MAE Calculation
- RGB Image Support
- Simple Graphical User Interface (GUI)

---

## Encryption Pipeline

```text
Original Image
      │
      ▼
Password-Based Key Generation
      │
      ▼
GTSMn Matrix Encryption
      │
      ▼
Shuffle Permutation
      │
      ▼
Henon Map Chaotic Diffusion
      │
      ▼
Encrypted Image
```

---

## Decryption Pipeline

```text
Encrypted Image
      │
      ▼
Inverse Henon Diffusion
      │
      ▼
Inverse Shuffle
      │
      ▼
Inverse GTSMn Matrix
      │
      ▼
Recovered Image
```

---

## Security Evaluation

The application includes several image encryption evaluation metrics:

- Histogram Analysis
- Entropy
- Adjacent Pixel Correlation
  - Horizontal
  - Vertical
  - Diagonal
- NPCR (Number of Pixels Change Rate)
- UACI (Unified Average Changing Intensity)
- MAE (Mean Absolute Error)

---

## Technologies

- Python 3.x
- NumPy
- Pillow (PIL)
- Matplotlib
- Pandas
- Numba
- Tkinter

---

## Project Structure

```text
Image-Encryption-GTSMn-Henon/
│
├── app.py
├── requirements.txt
├── README.md
│
├── core/
│   ├── encryption.py
│   ├── matrix.py
│   ├── metrics.py
│   ├── histogram_report.py
│   ├── security_report.py
│   └── preprocess.py
│
├── utils/
│   └── image_utils.py
│
├── assets/
│
├── output/
│
└── examples/
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/yourusername/image-encryption-gtsmn-henon.git
```

Move into the project directory

```bash
cd image-encryption-gtsmn-henon
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
python app.py
```

---

## Requirements

- Python 3.10+
- NumPy
- Pillow
- Matplotlib
- Pandas
- Numba

---

## Example Workflow

1. Open the application.
2. Select an RGB image.
3. Enter a password.
4. Perform image encryption.
5. Save the encrypted image.
6. Open the Decryption page.
7. Load the encrypted image.
8. Enter the same password.
9. Recover the original image.
10. View security analysis and evaluation metrics.

---

## Research Background

This project was developed as an undergraduate thesis in Information Systems. The proposed encryption scheme integrates:

- Matrix-based permutation using GTSMn
- Shuffle permutation
- Henon Map chaotic diffusion

to enhance image confidentiality while maintaining efficient encryption and decryption performance.

---

## Future Improvements

- Support for grayscale image encryption
- Additional chaotic maps
- Batch image encryption
- Performance optimization
- GPU acceleration
- More security evaluation metrics

---

## License

This project is intended for academic and research purposes.

---

## Author

**Rizki Putra Arindro**

Information Systems  
Faculty of Computer Science and Information Technology  
Gunadarma University

---

## Acknowledgements

Special thanks to the supervisors, researchers, and the open-source community whose contributions have supported the development of this project.
