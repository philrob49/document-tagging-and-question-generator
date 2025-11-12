"""
PDF Q&A Generator - Generate test questions/answers with meta-tags

Install required packages:
pip install pypdf2 anthropic

Usage:
python qa_generator.py input.pdf document_tags.csv output_qa.csv [expert_type]
"""

import re
import csv
import json
import io
import time
from typing import List, Dict
from collections import Counter
import PyPDF2
import anthropic
import os

class QAGenerator:
    def __init__(self, pdf_path: str, document_tags_csv: str, expert_type: str = "insurance expert", variations_per_question: int = 3, api_key: str = None):
        self.pdf_path = pdf_path
        self.document_tags_csv = document_tags_csv
        self.expert_type = expert_type
        self.variations_per_question = variations_per_question
        self.full_text = ""
        self.pages = []
        self.document_tags = []
        self.key_value_map = {}  # Maps key-values to their citations
        self.client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        
    def extract_text(self):
        """Extract ALL text from PDF"""
        print("Extracting document text...")
        try:
            with open(self.pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                all_text = []
                for page_num, page in enumerate(reader.pages, start=1):
                    text = page.extract_text()
                    self.pages.append({
                        'page': page_num,
                        'text': text
                    })
                    all_text.append(f"--- PAGE {page_num} ---\n{text}\n")

                self.full_text = '\n'.join(all_text)
            print(f"Extracted {len(self.pages)} pages")
        except FileNotFoundError:
            raise FileNotFoundError(f"PDF file not found: {self.pdf_path}")
        except PyPDF2.errors.PdfReadError as e:
            raise ValueError(f"Invalid or corrupted PDF file: {e}")
        except Exception as e:
            raise RuntimeError(f"Error extracting PDF text: {e}")
    
    def load_document_tags(self):
        """Load the document tags from step1 CSV"""
        print(f"Loading document tags from {self.document_tags_csv}...")

        required_columns = {'key', 'value', 'page', 'section', 'quote', 'long_citation'}

        try:
            with open(self.document_tags_csv, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)

                # Check columns exist
                if not required_columns.issubset(set(reader.fieldnames or [])):
                    missing = required_columns - set(reader.fieldnames or [])
                    raise ValueError(f"CSV missing required columns: {missing}")

                for row in reader:
                    self.document_tags.append(row)

                    # Build map of key-value pairs to their info
                    kv = f"{row['key']}:{row['value']}"
                    if kv not in self.key_value_map:
                        self.key_value_map[kv] = []
                    self.key_value_map[kv].append({
                        'page': row['page'],
                        'section': row['section'],
                        'quote': row['quote'],
                        'long_citation': row['long_citation']
                    })

            print(f"Loaded {len(self.document_tags)} document tags")
            print(f"Unique key-value pairs: {len(self.key_value_map)}")

            # Show distribution
            keys = [tag['key'] for tag in self.document_tags]
            key_counts = Counter(keys)
            print("\nKey distribution:")
            for key, count in sorted(key_counts.items()):
                print(f"  {key}: {count} values")

        except FileNotFoundError:
            raise FileNotFoundError(f"Tags CSV file not found: {self.document_tags_csv}")
        except csv.Error as e:
            raise ValueError(f"Invalid CSV format: {e}")
        except KeyError as e:
            raise ValueError(f"CSV row missing required field: {e}")
    
    def generate_qa_batch(self, iteration: int, batch_size: int = 30) -> str:
        """Generate a batch of Q&A pairs that cover the document tags"""
        # Adjust batch size for variations
        base_questions = batch_size // self.variations_per_question
        if base_questions < 1:
            base_questions = 1

        print(f"\nIteration {iteration}: Generating {base_questions} base questions with {self.variations_per_question} variations each...")

        # Token warning for large documents
        estimated_tokens = len(self.full_text) // 4
        if estimated_tokens > 150000:
            print(f"Warning: Document is very large (~{estimated_tokens} tokens)")
            print("This may cause API errors or high costs. Consider using a smaller document.")

        # Build tag summary for Claude
        tag_summary = {}
        for tag in self.document_tags:
            key = tag['key']
            if key not in tag_summary:
                tag_summary[key] = []
            tag_summary[key].append(tag['value'])

        tag_list = "\n".join([f"  {key}: {', '.join(values[:10])}" for key, values in tag_summary.items()])

        prompt = f"""You are a {self.expert_type} creating test questions and answers for this document.

COMPLETE DOCUMENT:
{self.full_text}

DOCUMENT TAGS (key-value pairs that categorize this document):
{tag_list}

CRITICAL TASK: Generate {base_questions} UNIQUE base topics, then create {self.variations_per_question} DIFFERENT WAYS to ask about each topic.

VARIATION REQUIREMENTS:
- Each base topic gets {self.variations_per_question} separate CSV rows
- Each row must have a DIFFERENT question phrasing
- Use different word order, synonyms, formality levels
- All variations of the same topic should have similar answers and tags

EXAMPLE - 3 variations of the same topic (theft coverage):
Row 1: "What happens if my car is stolen?","If your vehicle is stolen...","P.3 - insurance.pdf",...
Row 2: "If someone steals my vehicle, what am I covered for?","When your car is stolen...","P.3 - insurance.pdf",...
Row 3: "Does this policy cover car theft?","Yes, theft of your vehicle is covered...","P.3 - insurance.pdf",...

TAGGING REQUIREMENTS (CRITICAL):
- Tag EVERY question with ALL relevant key-value pairs (not just one!)
- Most questions relate to 2-5 different keys
- Example: A theft question should include: coverage (theft), claims (notification_timeframe), excess (theft_excess), etc.
- Be comprehensive - include ALL applicable tags from the list above

Return CSV format with columns:
question,answer,citation,sbert_ranked_tags,meta_tags

You MUST generate exactly {base_questions * self.variations_per_question} rows total.
That's {base_questions} topics with {self.variations_per_question} variations each.

Output CSV format directly (no markdown, no explanation)."""

        # API call with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=16000,
                    messages=[{"role": "user", "content": prompt}]
                )
                break
            except anthropic.RateLimitError as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"Rate limited. Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                else:
                    print("Rate limit exceeded after all retries")
                    raise
            except anthropic.APIError as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"API error: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"API error after all retries: {e}")
                    raise
            except Exception as e:
                print(f"Unexpected error during API call: {e}")
                raise
        
        csv_content = response.content[0].text.strip()
        
        # Remove markdown if present
        if "```" in csv_content:
            parts = csv_content.split("```")
            for i, part in enumerate(parts):
                if "csv" in part.lower() or (i > 0 and i < len(parts)-1):
                    csv_content = part.replace("csv", "").strip()
                    break
        
        return csv_content
    
    def parse_qa_csv(self, csv_content: str) -> List[Dict]:
        """Parse the Q&A CSV response using Python's csv module"""
        qa_pairs = []

        try:
            # Use csv.DictReader for robust CSV parsing
            reader = csv.DictReader(io.StringIO(csv_content))

            for row in reader:
                # Check if all required columns are present
                if all(key in row for key in ['question', 'answer', 'citation', 'sbert_ranked_tags', 'meta_tags']):
                    qa_pairs.append({
                        'question': row['question'],
                        'answer': row['answer'],
                        'citation': row['citation'],
                        'sbert_ranked_tags': row['sbert_ranked_tags'],
                        'meta_tags': row['meta_tags']
                    })
                else:
                    print(f"Warning: Skipping row with missing columns")

        except csv.Error as e:
            print(f"Warning: CSV parsing error: {e}")
        except Exception as e:
            print(f"Warning: Unexpected error parsing CSV: {e}")

        return qa_pairs
    
    def validate_and_clean_tags(self, qa_pairs: List[Dict]) -> List[Dict]:
        """Ensure all tags in Q&A match actual document tags"""
        print("\nValidating and cleaning meta-tags...")
        
        valid_keys = set(tag['key'] for tag in self.document_tags)
        valid_kv_pairs = set(f"{tag['key']}:{tag['value']}" for tag in self.document_tags)
        
        cleaned = []
        for idx, qa in enumerate(qa_pairs):
            try:
                # Clean up meta_tags - remove outer quotes if double-quoted
                meta_str = qa['meta_tags'].strip()
                if meta_str.startswith('""') and meta_str.endswith('""'):
                    meta_str = meta_str[1:-1]
                
                # Replace escaped quotes
                meta_str = meta_str.replace('""', '"')
                
                # Parse meta_tags JSON
                meta_tags = json.loads(meta_str)
                
                # Filter to only valid keys and values
                cleaned_meta = {}
                for key, values in meta_tags.items():
                    if key in valid_keys and key != 'source_file':
                        # Only keep values that exist in document tags
                        valid_values = [v for v in values if f"{key}:{v}" in valid_kv_pairs]
                        if valid_values:
                            cleaned_meta[key] = valid_values
                
                # Add source_file if present
                if 'source_file' in meta_tags:
                    cleaned_meta['source_file'] = meta_tags['source_file']
                
                # Update the meta_tags
                qa['meta_tags'] = json.dumps(cleaned_meta)
                cleaned.append(qa)
                
            except json.JSONDecodeError as e:
                if idx < 3:  # Only show first 3 errors for debugging
                    print(f"Warning: Invalid JSON in meta_tags (row {idx}): {str(e)}")
                    print(f"  Raw meta_tags: {qa['meta_tags'][:100]}...")
                continue
            except Exception as e:
                if idx < 3:
                    print(f"Warning: Error processing row {idx}: {str(e)}")
                continue
        
        print(f"Validated {len(cleaned)}/{len(qa_pairs)} Q&A pairs")
        return cleaned
    
    def process(self, output_csv: str, target_questions: int = 100):
        """Generate Q&A pairs until target reached"""
        print(f"Generating Q&A pairs for {self.pdf_path}")
        print(f"Target: {target_questions} questions\n")
        
        self.extract_text()
        self.load_document_tags()
        
        all_qa_pairs = []
        iteration = 1
        batch_size = 30
        
        while len(all_qa_pairs) < target_questions:
            remaining = target_questions - len(all_qa_pairs)
            current_batch_size = min(batch_size, remaining + 5)  # Generate a few extra
            
            print(f"\n{'='*70}")
            print(f"BATCH {iteration} (Target: {current_batch_size} questions)")
            print(f"{'='*70}")
            
            # Generate Q&A batch
            csv_content = self.generate_qa_batch(iteration, current_batch_size)
            
            # Parse and validate
            qa_batch = self.parse_qa_csv(csv_content)
            qa_batch = self.validate_and_clean_tags(qa_batch)
            
            all_qa_pairs.extend(qa_batch)
            
            print(f"Generated: {len(qa_batch)} valid Q&A pairs")
            print(f"Total so far: {len(all_qa_pairs)}/{target_questions}")
            
            if len(all_qa_pairs) >= target_questions:
                break
            
            iteration += 1
            
            if iteration > 10:  # Safety limit
                print("\nReached iteration limit")
                break
        
        # Trim to exact target
        all_qa_pairs = all_qa_pairs[:target_questions]

        # Write to CSV
        print(f"\nWriting {len(all_qa_pairs)} Q&A pairs to {output_csv}...")
        try:
            with open(output_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['question', 'answer', 'citation', 'sbert_ranked_tags', 'meta_tags'])
                writer.writeheader()
                writer.writerows(all_qa_pairs)
            print(f"Successfully wrote output to {output_csv}")
        except IOError as e:
            raise IOError(f"Failed to write output file: {e}")
        except Exception as e:
            raise RuntimeError(f"Error writing CSV file: {e}")
        
        # Summary
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"Total Q&A pairs generated: {len(all_qa_pairs)}")
        
        # Tag usage stats
        tag_usage = {}
        for qa in all_qa_pairs:
            try:
                meta = json.loads(qa['meta_tags'])
                for key in meta.keys():
                    if key != 'source_file':
                        tag_usage[key] = tag_usage.get(key, 0) + 1
            except json.JSONDecodeError:
                # Skip rows with invalid JSON
                continue
            except Exception:
                # Skip any other errors
                continue

        print("\nTag usage in questions:")
        for key, count in sorted(tag_usage.items()):
            print(f"  {key}: {count} questions")
        print("="*70)

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 4:
        print("Usage: python qa_generator.py input.pdf document_tags.csv output_qa.csv [expert_type] [num_questions] [variations]")
        print("\nParameters:")
        print("  variations: Number of ways to ask each question (default: 3)")
        print("\nExamples:")
        print("  python qa_generator.py policy.pdf tags.csv qa_output.csv")
        print("  python qa_generator.py policy.pdf tags.csv qa_output.csv 'insurance expert' 50")
        print("  python qa_generator.py medical.pdf tags.csv qa_output.csv 'medical professional' 60 5")
        sys.exit(1)

    # Parse arguments
    pdf_path = sys.argv[1]
    tags_csv = sys.argv[2]
    output_csv = sys.argv[3]
    expert_type = sys.argv[4] if len(sys.argv) > 4 else "insurance expert"

    # Validate file paths exist
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found: {pdf_path}")
        sys.exit(1)
    if not os.path.exists(tags_csv):
        print(f"Error: Tags CSV file not found: {tags_csv}")
        sys.exit(1)

    # Parse and validate num_questions
    try:
        num_questions = int(sys.argv[5]) if len(sys.argv) > 5 else 100
        if num_questions <= 0:
            print("Error: num_questions must be a positive integer")
            sys.exit(1)
    except ValueError:
        print(f"Error: num_questions must be an integer, got: {sys.argv[5]}")
        sys.exit(1)

    # Parse and validate variations
    try:
        variations = int(sys.argv[6]) if len(sys.argv) > 6 else 3
        if variations <= 0:
            print("Error: variations must be a positive integer")
            sys.exit(1)
    except ValueError:
        print(f"Error: variations must be an integer, got: {sys.argv[6]}")
        sys.exit(1)

    # Create generator and process
    try:
        generator = QAGenerator(pdf_path, tags_csv, expert_type=expert_type, variations_per_question=variations)
        generator.process(output_csv, target_questions=num_questions)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)