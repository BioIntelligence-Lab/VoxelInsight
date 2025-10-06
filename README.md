# 🧠 VoxelInsight

**VoxelInsight** is a conversational AI assistant for biomedical imaging that bridges large-scale data repositories and advanced image analysis tools — all through natural language. Built with Chainlit and OpenAI’s GPT-5, VoxelInsight turns plain English into powerful radiology workflows.

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
- **Segmentation**: Automatically segment organs, lesions, or tumors using **TotalSegmentator** and **MONAI**
- **Radiomics**: Extract texture, shape, and first-order features using **PyRadiomics**
- **Clinical Modeling**: Train models to predict clinical endpoints
- Supports **DICOM** and **NIfTI** inputs

---

## ⚙️ Installation & Setup

### 0) **Requirements**
- macOS (Apple Silicon) or Linux
- Python 3.10
- Docker (for PostgreSQL + S3)
- (macOS only) Colima may be required for Docker compatibility

### 1) **Clone the repository**:

``` bash
git clone https://github.com/BioIntelligence-Lab/VoxelInsight.git
cd voxelinsight
```

### 2) **Create & activate a clean environment**:

``` bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
```

If you prefer conda use this:

``` bash
conda create -n voxelinsight python=3.10 -y
conda activate voxelinsight
python -m pip install -U pip setuptools wheel
python -m pip install "numpy<2.0" "SimpleITK>=2.2"
python -m pip install "pyradiomics==3.0.1" --no-build-isolation
python -m pip install -r requirements.txt
```

### 3) **Install dependencies**

``` bash
python -m pip install "numpy<2.0" "SimpleITK>=2.2"
python -m pip install "pyradiomics==3.0.1" --no-build-isolation
pip install -r requirements.txt
``` 

### 4) **Setup Environment**

Add your keys to the .env file

``` bash
# OpenAI / model keys
OPENAI_API_KEY=sk-...

# Chainlit auth (GitHub OAuth)
OAUTH_GITHUB_CLIENT_ID=...
OAUTH_GITHUB_CLIENT_SECRET=...
CHAINLIT_AUTH_SECRET=some-long-random-string

# Optional: MIDRC credential file for downloads
MIDRC_CRED=/absolute/path/to/midrc_credentials.json

# Database (preset)
DATABASE_URL=postgresql://root:root@localhost:5432/postgres
```

### How to obtain keys
#### OAUTH_GITHUB_CLIENT_ID / OAUTH_GITHUB_CLIENT_SECRET:
------
Go to GitHub [Developer Settings → OAuth Apps](https://github.com/settings/developers)

Create a new Oauth App.

  - Homepage URL: http://localhost:8000
  - Callback URL: http://localhost:8000/auth/callback

Copy the Client ID and Client Secret into .env.

#### CHAINLIT_AUTH_SECRET
-----
Any long random string (e.g., openssl rand -hex 32).

#### MIDRC_CRED (optional for MIDRC downloads):
------
Obtain credentials from MIDRC:
  - Go to [MIDRC](https://www.midrc.org/data-launch-page).
  - Click option "Centralized data: MIDRC Data Commons".
  - Create an account and log in (top right corner).
  - Click profile icon in top right and select "View Profile".
  - Create a new API key and download JSON file.

Save the JSON file locally and point MIDRC_CRED to its absolute path. 

### 5) **Setup a local database**

The defaul DATABASE_URL points to:

``` bash
postgresql://root:root@localhost:5432/postgres
```

To initialize this database:

``` bash
# Start PostgreSQL + S3 services
docker compose up -d

# Apply Prisma schema migrations (once per machine)
chainlit datalayer migrate
```
On macOS, you may need to run Docker with Colima

### 6) **Run the Application**
``` bash
chainlit run app.py
```  

To select a different port:

``` bash
chainlit run app.py -p 3001
```

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
  Support for additional pretrained models on top of **TotalSegmentator**, including tumor and disease-specific segmentations.

- 🧱 **Foundation Model Integration**  
  Plug-and-play with leading foundation models for medical imaging (e.g., BioMedCLIP, MERLIN) to enhance embedding-based retrieval and classification.

- 🔄 **Longitudinal Imaging Analysis**  
  Track changes across timepoints using embeddings, volumes, and derived biomarkers to study treatment response or disease progression.

- 📊 **Quantitative Imaging Reports**  
  Export structured reports summarizing volumetric, radiomic, and anatomical measurements from any imaging study.

- 🧪 **Interactive Visualization Tools**  
  Scroll, overlay, and compare segmentations directly within the chat environment.

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

**How to contribute a new tool:**

1. Create a new file in tools/, e.g. tools/my_tool.py
2. Define args schema with Pydantic for clean API
3. Implement the agent class with an async run(task, state) method
4. Wrap with toolify_agent so the supervisor can call it
5. Configure it at startup in app.py via configure_my_tool()
6. Add description so the LLM supervisor knows when to use it

For working references check tools/

**Example Tool Stub:**

```python
class MyToolAgent:
    name = "your_tool"
    model = None

    async def run(self, task: Task, state: ConversationState) -> TaskResult:
        arg1 = task.kwargs.get("arg1")
        # Tool Functionality…
        return TaskResult(output={"text": f"Processed {arg1}"})
```

register it with

```python
@toolify_agent(
    name="your_tool",
    description="Describe your tool’s function here.",
    args_schema=MyToolArgs,
)
async def my_tool_runner(...):
    # call agent.run()
```


