# pdf_extractor.py
import json
import os
from pathlib import Path
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import PdfFormatOption

# Configure pipeline options
pipeline_options = PdfPipelineOptions()
pipeline_options.do_ocr = False # True for scanned PDFs
pipeline_options.do_table_structure = True # Extract tables

# Initialize the converter
converter = DocumentConverter(
format_options={
InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
})
# Set PDF path
pdf_path = Path("/home/ubuntu/insurance_lab/lab3/Sample_Pdfs/health_policy.pdf")
print(f"Processing: {pdf_path.name}")
result = converter.convert(str(pdf_path))

# Get the Docling document object
doc = result.document

# Export to Markdown (preserves headings, tables, lists)
markdown_output = doc.export_to_markdown()
print("\n=== Markdown Output ===")
print(markdown_output)

# Save markdown to file
with open("extracted_policy.md", "w") as f:
    f.write(markdown_output)
    print("\nSaved to extracted_policy.md")

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv(dotenv_path="/home/ubuntu/insurance_lab/.env")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

extraction_prompt = ChatPromptTemplate.from_messages([
("system",
"Extract structured data from this insurance policy. "
"Return JSON with keys: policy_name, sum_insured, exclusions[], "
"benefits[], waiting_periods[]. No extra text, just JSON."),
("human", "Policy Document:\n{document}"),
])
chain = extraction_prompt | llm | StrOutputParser()
result_json = chain.invoke({"document": markdown_output})
parsed = json.loads(result_json)
print("\n=== Extracted Structure ===")
print(json.dumps(parsed, indent=2))
