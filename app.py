from openai import AsyncOpenAI
import chainlit as cl
import re
import io
import base64
import matplotlib.pyplot as plt
import pandas as pd
import pydicom
import os
from pathlib import Path
import tempfile
import shutil


client = AsyncOpenAI()

# Instrument the OpenAI client
cl.instrument_openai()

settings = {
    "model": "gpt-4o",
    "temperature": 0,
    # ... more settings
}

from idc_index import index

IDC_Client = index.IDCClient()
df_IDC = IDC_Client.index
#df_BIH = pd.read_csv("Data/midrc_distributed_subjects.csv")
#df_MIDRC = pd.read_csv("Data/MIDRC_Cases_table.csv")
df_BIH = pd.DataFrame()
df_MIDRC = pd.DataFrame()
def parse(chat_response):
  #print('H1')
  #code_blocks = re.findall(r"```sql(.+?)```", chat_response, re.DOTALL)
  code_blocks = re.findall(r"```(.+?)```", chat_response, re.DOTALL)
  if len(code_blocks)>0:
    code_blocks[0] = code_blocks[0].replace('python','')
    code_blocks[0] = code_blocks[0].replace(';','')
    code_blocks[0] = code_blocks[0]
  return code_blocks


with open("Data/TotalSegmentatorMappingsCT.tsx", "r", encoding="utf-8") as f:
    ct_mappings = f.read()


@cl.on_message
async def on_message(message: cl.Message):
    file_elements = [
        el for el in (message.elements or [])
        if isinstance(el, cl.File)
    ]

    files_info = []
    for f in file_elements:
        # f has: path, name, mime, size (bytes)
        '''
        files_info.append(
            {"path": f.path, "name": f.name, "mime": f.mime, "size": f.size}
        )
        '''
        tmpdir = Path(tempfile.mkdtemp())
        new_path = tmpdir / f.name
        shutil.copy(f.path, new_path)
        files_info.append(new_path)

    #file_paths = [f["path"] for f in files_info]

    print(files_info)

    pretext= f'Please be as specific as possible and only return the final python code enclosed in ```. \
    Do not provide explanations. You are a medical imaging research assistant.\
    Your name is VoxelInsight. You are part of BioInsight\
    If users upload a file, the file paths will be listed here: {files_info} \
    You have the following key capabilities right now.\
    (1) User can interact with public and private datasets with natural language and ask questions like how many collections are present in X\
    (2) User can do standard image processing like nifti conversion, image registration, etc.\
    (3) User can segment medical imaging datasets\
    (4) User can extract imaging biomarkers using radiomics\
    The details about all the capabilities are as follows\
    I have created two dataframes \
    the first dataframe contains all the data for the platform IDC and is created using the package idc-index. \
    This data frame was created using the following command: IDC_Client = index.IDCClient(); df_IDC = IDC_Client.index. Assume df_IDC is already present. \
    Using the pandas dataframe, df_IDC which has data fields such as: \
    collection_id: id of different collections or datasets on IDC,\
    Modality: modality of the images or imaging datasets (e.g., CT, MR, PT (for PET), etc.). Make sure to use MR when the user asks for MRI, \
    BodyPartExamined: body part examined (for example, brain or lung, etc.), \
    SeriesDescription: Different series or sequences contained within a dataset (e.g., MR contains DWI, T1WI, etc.), \
    PatientID: ID of different patients, PatientSex: Sex of different patients (e.g., M for male), \
    PatientAge: Age of different patients (Hint: Use SAFE_CAST), \
    Manufacturer: Scanner manufacturer, \
    ManufacturerModelName: Name of the scanner model, \
    instanceCount: Numbre of images in the series \
    StudyDescription: Description of the studies, \
    SeriesInstanceUID: Series IDs, \
    These were the commonly queried data fields. There are other data fields as well. \
    Use your best guess if the user asks for information present outside the provided data fields. \
    Make sure the queries are case insensitive and use Regex wherever necessary. \
    If the user wants to download data, use the following commands: from idc_index import index; IDC_Client = index.IDCClient(); IDC_Client.download_from_selection(seriesInstanceUID=selected_series, downloadDir=".") \
    The second dataframe df_BIH contains the multi-source indexed dataframe\
    This index contains study level data from multiple source or public platforms including IDC, MIDRC, TCIA, among others\
    You have to identify whether the user wants to query this BDF dataframe or the IDC dataframe and answer their query\
    The df_BIH dataframe has the following fields\
    subject_id: patient id \
    commons_name: name of the data source like IDC, MIDRC, AIMI\
    metadata_source_version: version of the metadata \
    race: race of the patient\
    disease_type: disease type of the patient \
    data_url_doi: url to access the data, sometimes this points to a journal\
    StudyDescription: Description of what the study contains\
    StudyInstanceUID: instance UID to look at this particular study\
    study_viewer_url: link to OHIF viewer that hosts the dataset\
    collection_id: id of different collections or datasets\
    license: licens whether data is public or not etc.\
    primary_site: primary body site for which data is collected \
    metadata_source_date: date metadata was sourced\
    commons_long_name: long name of the data source\
    PatientAge: Age of the patient - numeric value\
    EthnicGroup: Ethnic group\
    PatientSex: Sex of the patient\
    collection_id: id of the collection\
    You also have access to the following tools if you are working with local data where the user provides path to the data: \
    1) DICOM to NIfTI conversion using the dicom2nifti Python package. \
    2) Image visualization using ipywidgets and matplotlib for viewing DICOM and NIfTI images. \
    3) Segmentation using TotalSegmentator, which supports organ and lesion segmentation from CT/MRI NIfTI files. \
    Example Command line usage: TotalSegmentator -i ct.nii.gz -o segmentations -ta <task_name> -rs <roi_subset>\
    example for normal tissue: TotalSegmentator -i ct.nii.gz -o seg -ta total -rs liver\
    Refer to the following file for roi_subset to task_name mappings: {ct_mappings}. If you want all the regions under a given task_name, do not refer to a specific roi_subset. \
    When a segmentation is saved, the path will be in this format: <output_segmentation_directory>/<roi_subset>.nii.gz. \
    If multiple roi_subsets are used, there will be one file in the output segmentation directory for each roi_subset. \
    For segmenting the lung, do not use task name lung_nodules, instead use task name total. \
    nnunet: currently supports brain tumor segmentation\
    Here is an example command if the user asks to perform brain tumor segmentation. The task number is 501\
    nnUNetv2_predict -i INPUT_DATA_PATH -o OUTPUT_DATA_PATH -d 501 -c 3d_fullres -f 0\
    4) Radiomics extraction using PyRadiomics, which computes shape, first-order, and texture features from segmented regions. \
    Usage Command line: pyradiomics <path/to/image> <path/to/segmentation> -o results.csv -f csv\
    When a user asks a clinical imaging question (e.g., “What is the liver volume in this scan?”), you should: \
    Run TotalSegmentator on the input NIfTI file to segment the requested region (e.g., liver).\
    Use PyRadiomics to extract relevant metrics from the segmentation. \
    Return the answer (e.g., volume in cc).\
    Use your best guess if the user asks for information present outside the provided data fields. \
    Make sure the queries are case insensitive and use Regex wherever necessary. Write a python script\
    to do the following and store the final output to a variable called res_query.\
    Make sure to store the final result in a variable called `res_query`. This result can be:\
    - A **pandas DataFrame** (for tabular results).\
    - A **Python scalar** (e.g., int, float, or string).\
    - A **matplotlib figure**, in which case store a BytesIO object containing the saved PNG image of the figure in `res_query`.\
    - Example: use `buf = io.BytesIO(); fig.savefig(buf, format="png"); buf.seek(0); res_query = buf`\
    \
    Ensure any figure-based outputs also call plt.close(fig) after saving to avoid memory leaks.\
    Always ensure that `res_query` contains the final displayable or usable result. Do not print or display the result outside of storing it in `res_query`.\
    '
    prompt = message.content
    messages = []
    messages.append({"role": "user", "content": pretext+prompt})
    completion = await client.chat.completions.create(
    messages = messages,
    **settings
    )
    chat_response = completion.choices[0].message.content
    query = parse(chat_response)
    q = query[0]
    print(q)
    # exec(q)
    

    # Create a local namespace
    #local_vars = {"df_IDC": df_IDC,"df_BIH": df_BIH, "pydicom":pydicom,"os":os}
    # Add everything you might use to local_vars
    local_vars = {
    "os": os,
    "pydicom": pydicom,
    "plt": plt,
    "io": io,
    "df_IDC": df_IDC,
    "df_BIH": df_BIH
    }

    
    await cl.Message(content=f"I have generated the following python code in response to your query:\n```python\n{q}\n```").send()
    await cl.Message(content="# Result: \n").send()
    exec(q, local_vars, local_vars)
    #exec(q, globals(), globals())

    # Now access the variable
    res_query = local_vars.get("res_query")
    print(res_query)

    

    if res_query is None:
            await cl.Message(content="No result found in res_query.").send()
            return

    # If it's a matplotlib figure saved as BytesIO
    if isinstance(res_query, io.BytesIO):
        res_query.seek(0)
        await cl.Message(content="Here is your plot:", elements=[
        cl.Image(name="plot.png", mime="image/png", content=res_query.read())
        ]).send()

    # If it's a base64-encoded image string
    elif isinstance(res_query, str) and res_query.startswith("data:image/png;base64,"):
        await cl.Message(content=f"![plot]({res_query})").send()



    # # If it's a pandas DataFrame
    # elif isinstance(res_query, pd.DataFrame):
    #     await cl.Message(content=res_query.head().to_markdown(index=False)).send()

    # # If it's a scalar or string
    # elif isinstance(res_query, (int, float, str)):
    #     await cl.Message(content=str(res_query)).send()

    else:
        #await cl.Message(content=f"res_query is of unsupported type: {type(res_query)}").send()
        await cl.Message(content=res_query).send()
    # await cl.Message(content=res_query).send()