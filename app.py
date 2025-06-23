from openai import AsyncOpenAI
import chainlit as cl
import re
import io
import base64
import matplotlib.pyplot as plt
import pandas as pd
import pydicom
import os


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
df_MIDRC = pd.read_csv("C:/Users/vpare/OneDrive - UTHealth Houston/Research/ARPA_H/midrc_distributed_subjects.csv")
def parse(chat_response):
  #print('H1')
  #code_blocks = re.findall(r"```sql(.+?)```", chat_response, re.DOTALL)
  code_blocks = re.findall(r"```(.+?)```", chat_response, re.DOTALL)
  if len(code_blocks)>0:
    code_blocks[0] = code_blocks[0].replace('python','')
    code_blocks[0] = code_blocks[0].replace(';','')
    code_blocks[0] = code_blocks[0]
  return code_blocks





@cl.on_message
async def on_message(message: cl.Message):
    
    pretext= 'Please be as specific as possible and only return the final python code enclosed in ```. \
    Do not provide explanations. I have created two dataframes \
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
    The second dataframe df_MIDRC contains the multi-source indexed dataframe\
    This index contains study level data from multiple source or public platforms including IDC, MIDRC, TCIA, among others\
    You have to identify whether the user wants to query this BDF dataframe or the IDC dataframe and answer their query\
    The MIDC dataframe has the following fields\
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
    For tumor, the task is different: here is an example: TotalSegmentator -i ct.nii.gz -o seg -ta liver_vessels\
    Here seg is the folder name\
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
    #local_vars = {"df_IDC": df_IDC,"df_MIDRC": df_MIDRC, "pydicom":pydicom,"os":os}
    # Add everything you might use to local_vars
    local_vars = {
    "os": os,
    "pydicom": pydicom,
    "plt": plt,
    "io": io,
    "df_IDC": df_IDC,
    "df_MIDRC": df_MIDRC
    }
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