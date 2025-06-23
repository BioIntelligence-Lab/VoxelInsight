# 🧠 VoxelInsight

**VoxelInsight** is a conversational AI assistant for biomedical imaging that bridges large-scale data repositories and advanced image analysis tools — all through natural language. Built with Chainlit and OpenAI’s GPT-4o, VoxelInsight turns plain English into powerful radiology workflows.

---

## 🚀 Key Features

### 🔍 Natural Language Querying of Imaging Repositories
- Search and explore datasets from platforms like **MIDRC**, **IDC**, and **TCIA** using **plain English prompts**
- Indexed metadata includes:
  - Body part examined
  - Imaging modality (CT, MR, PET, etc.)
  - Study and series descriptions
  - Scanner manufacturer and model
  - Patient demographics and more
- Example queries:
  - “Which collections contain liver CT data?”
  - “Create a bar chart of patient counts per MIDRC collection”
  - “How many MRI scanners were used in the UPenn GBM dataset?”

### 🧠 AI-Powered Imaging Analysis
- **Segmentation**: Automatically segment organs, lesions, or tumors using **TotalSegmentator**
- **Radiomics**: Extract texture, shape, and first-order features using **PyRadiomics**
- **Clinical Modeling**: Train models to predict clinical endpoints
- Supports **DICOM** and **NIfTI** inputs

### 🗃️ Dual-Index Backend
- `df_IDC`: Real-time access to the Imaging Data Commons
- `df_MIDRC`: A harmonized study-level index of MIDRC, TCIA, AIMI, and other sources

---

## ⚙️ Installation & Setup

### 🐍 Requirements
- Python 3.8 or higher
- pip (Python package installer)

### 📥 Step-by-Step Installation

1. **Clone the repository**:
   
   git clone https://github.com/your-username/voxelinsight.git
   cd voxelinsight

2. **Install dependencies**

    pip install -r requirements.txt
  
3. **Setup Chainlit environment variables**

   Create a file named .env in the same folder as your app.py file. Add your OpenAI API key in the    OPENAI_API_KEY variable.
   
4. **Run the Application**

   chainlit run app.py -w
   
---

## 🧪 Example Prompts

Some example questions you can ask VoxelInsight:
- Which platforms contain COVID-19 data?
- List the collections on the MIDRC platform.
- How many patients are in the CheXpert dataset on AIMI?
- Segment the liver from this CT scan and give me its volume.
- Segment brain tumors from all patients in the upenn_gbm collection, extract radiomics, and train a MLP classifier to predict overall survival

## 🧭 Roadmap & Upcoming Features

VoxelInsight is continuously expanding its imaging intelligence. Upcoming features include:

- 🧠 **Expanded Model Library**  
  Support for additional pretrained models on top of **TotalSegmentator**, including tumor subtyping, lesion characterization, and disease-specific segmentations.

- 🧱 **Foundation Model Integration**  
  Plug-and-play with leading foundation models for medical imaging (e.g., BioMedCLIP, MERLIN, nnDetection) to enhance embedding-based retrieval and classification.

- 🔄 **Longitudinal Imaging Analysis**  
  Track changes across timepoints using embeddings, volumes, and derived biomarkers to study treatment response or disease progression.

- 📊 **Quantitative Imaging Reports**  
  Export structured reports summarizing volumetric, radiomic, and anatomical measurements from any imaging study.

- 🧪 **Interactive Visualization Tools**  
  Scroll, overlay, and compare segmentations directly within the chat environment (coming soon to the UI).

---

## 🤝 Contributing

We welcome contributions including:
- 🧩 New segmentation or model integrations
- 🧪 Visualization and analysis tools
- 📚 Dataset plugins or indexing enhancements
- 🛠️ Documentation and usability improvements

**How to contribute:**
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with a clear description of your changes

For major features or ideas, please open an issue to start a discussion.  
Let’s shape the future of imaging AI together!
