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

Using Python Virtual Environment

``` bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
```

Using Anaconda

``` bash
conda create -n voxelinsight python=3.10 -y
conda activate voxelinsight
python -m pip install -U pip setuptools wheel
```

### 3) **Install dependencies**

``` bash
python -m pip install "numpy<2.0" "SimpleITK>=2.2"
python -m pip install "pyradiomics==3.0.1" --no-build-isolation
pip install -r requirements.txt
```

### 4) **Set up the environment (.env file)**

We have developed our application to work with GPT APIs. This requires adding your OpenAI keys to the .env file

``` bash
# OpenAI / model keys
OPENAI_API_KEY=sk-...
```

If you are interested in using different integrations, including locally hosted LLMs, check out https://docs.chainlit.io/integrations/

### 5) **Run the Application**
``` bash
chainlit run app.py
```  

To select a different port:

``` bash
chainlit run app.py -p 3001
```

### Note: (Optional) The next steps (Step 6 and Step 7) are optional and required if you want to set up a database to store user interactions.
### if you do not plan to set this up, please comment out lines 44-51 in app.py (shown below)

```
# @cl.oauth_callback
# def oauth_callback(
#     provider_id: str,
#     token: str,
#     raw_user_data: Dict[str, str],
#     default_user: cl.User,
# ) -> Optional[cl.User]:
#     return default_user
```


### 6) **Set up a local database (Optional)**

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

Once done, add the following to your .env file

``` bash
# Database (preset)
DATABASE_URL=postgresql://root:root@localhost:5432/postgres
```



### 7) **Set up user authentication (Optional)**

Your application will be public by default without any controlled access. In addition, the user interactions would not be stored. 
If you want your application to either be private or store user interactions, follow the instructions at https://docs.chainlit.io/authentication/overview to set up your authentication. 

For example, if you want to setup Google OAuth, you can follow the instructions here: https://docs.chainlit.io/authentication/oauth#google
Once done, you will need to set the following environment variables (i.e. add the following lines to the .env file):
``` bash
- OAUTH_GOOGLE_CLIENT_ID: Client ID
- OAUTH_GOOGLE_CLIENT_SECRET: Client secret
- CHAINLIT_AUTH_SECRET: Chainlit secret
```

**Note: Chainlit secret key is required for any authentication method and can be generated using**
```
chainlit create-secret
```

---

## Setting up VoxelInsight Agents

### VoxelInsight currently supports the following agents (with setup instructions)

### Data Agents
#### 1) IDC Agent 
The IDC agent allows users to interact with the "Imaging Data Commons". This agent is already set up and ready to use. The IDC agent has the following capabilities
   - General querying (e.g. How many collections on IDC contain brain data?)
   - Download (e.g., Can you download the FLAIR series for the patient UPENN-GBM-00144 from IDC?)
   - Statistical Visualization (e.g., Can you plot a histogram of the number of patients in each breast cancer collection on IDC?)
   - Image/Volume Visualization (e.g., Can you download the FLAIR series for the patient UPENN-GBM-00144 from IDC and visualize it on a slider?)
   
#### 2) TCIA Agent 
The TCIA agent allows users to interact with "The Cancer Imaging Archive". This agent uses BDF Imaging Hub for general querying and TCIA APIs for downloads. The download capability for this agent is set up by default. For querying, you will need the BIH csv file. Further instructions are provided in the MIDRC Agent setup. Similar to IDC, the TCIA agent can also query and download data from the TCIA. Some example capabilities and queries include
   - General (How many collections on TCIA contain liver data?)
   - Download (Download an example patient from the NLST collection on TCIA?)
   - Statistical Visualization (e.g., can you plot a pie chart depicting the sex distribution of the NLST collection?)
   - Image/Volume Visualization (e.g., can you download and visualize a CT scan from the NLST collection?)

#### 3) MIDRC Agent 
The MIDRC agent allows users to interact with the "MIDRC" database. This agent requires set up before it is ready to use. If you do not intend to use the MIDRC agent, please replace the following line from app.py. **Note: You will also not be able to perform TCIA queries (only be able to perform TCIA downloads without setting up the MIDRC agent). In the upcoming updates, we will remove this dependency**
```
df_MIDRC = pd.read_parquet("midrc_mirror/nodes/midrc_files_wide.parquet")
```
with
```
df_MIDRC = pd.DataFrame()
```

##### Step 1: MIDRC Credentials
   
Obtain credentials from MIDRC:
  - Go to [MIDRC](https://www.midrc.org/data-launch-page).
  - Click option "Centralized data: MIDRC Data Commons".
  - Create an account and log in (top right corner).
  - Click profile icon in top right and select "View Profile".
  - Create a new API key and download JSON file.

Save the JSON file locally and update the .env file to point MIDRC_CRED to its absolute path. 
```
MIDRC_CRED=/path/to/midrc_credentials.json
```

##### Step 2: Download the MIDRC parquet file by running the following command
```
python midrc_graph_mirror.py
```

### Image Processing and AI Agents (these will be set up automatically upon installation -- no additional steps required)

#### 1) Standard Image Processing 
- NIfTI: Convert any dicom folder to NIfTI format. Upload your dicom folder as a zip file and ask VoxelInsight to convert it to NIfTI.
- Visualization: Upload and visualize any image (Dicom or PNG/JPEG/etc.) or volume (Dicom or NiFTI)

#### 2) Segmentation
- **Total Segmentator**: You can call the primary total segmentator model or any of the sub models within the total segmentator framework to segment specific or all possible regions. The primary advantage of this agent is that you do not need to remember specific label names or the model a specific ROI is available in. Here are some examples (assuming you have uploaded a nifti file for each query):
  - Can you segment liver and spleen in the attached image and overlay the result?
  - Can you segment all lung lobes in the attached image and overlay the result?
  - Segment liver from this CT scan and give me its volume and entropy.
  - Can you segment liver tumor from all CT scans in the folder in the path: XYZ, extract radiomics and save a csv file with radiomic features for all CT scans. 
- **MONAI**: The VoxelInsight framework also allows performing segmentation using different models present in the MONAI model zoo. **Note: This agent is currently under development and not all models in MONAI model zoo may be available)**
  - Can you segment brain tumor in the attached nifti and overlay the result on the FLAIR sequence. (Note: This requires a 4D Nifti input)

#### 3) Radiomics: You can extract radiomics from any image-mask pair(s). This agent uses pyRadiomics for radiomics extraction.
- Extract radiomics from the uploaded image and mask.
- What is the volume and surface area of the liver in the given CT scan? (Note: This first runs totalsegmenttor to segment liver and then runs pyradiomics to get the volume and surface area)
---

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


