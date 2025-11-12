"""
PDF Document Tagger - Generic Expert Analysis

Install required packages:
pip install pypdf2 anthropic

Usage:
python pdf_tagger.py input.pdf output.csv [expert_type] [target_coverage]

Examples:
  python pdf_tagger.py policy.pdf output.csv "insurance expert"
  python pdf_tagger.py medical.pdf output.csv "medical professional" 70
  python pdf_tagger.py textbook.pdf output.csv "mathematics professor"
"""

import re
import csv
from collections import Counter
from typing import List, Dict
import PyPDF2
import anthropic
import os

class PDFTagger:
    def __init__(self, pdf_path: str, expert_type: str = "insurance expert", api_key: str = None):
        self.pdf_path = pdf_path
        self.expert_type = expert_type
        self.full_text = ""
        self.pages = []
        self.client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.pass_threshold = 0.75
        
    def extract_text(self):
        """Extract ALL text from PDF"""
        print("Extracting complete document text...")
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
        
        print(f"Extracted {len(self.pages)} pages ({len(self.full_text)} characters)")
    
    def analyze_document_as_expert(self, iteration: int, previous_tags: List[str] = None) -> str:
        """
        Have Claude read the ENTIRE document from the specified expert perspective
        and extract key topics with citations
        """
        print(f"\nIteration {iteration}: Claude analyzing document as {self.expert_type}...")
        
        # Build context from previous iteration
        previous_context = ""
        if previous_tags and iteration > 1:
            covered = ", ".join(previous_tags[:20])
            previous_context = f"""
Previous iteration covered these topics: {covered}...

IMPORTANT: Generate DIFFERENT topics this iteration. Focus on areas NOT yet covered.
Look deeper into the document for more specific details."""
        
        # Customize prompt based on expert type
        expert_lower = self.expert_type.lower()
        
        if "insurance" in expert_lower:
            doc_type = "insurance document"
            customer_focus = "customers inquiring about insurance"
            example_keys = "coverage,exclusions,claims_process,premiums,deductibles,limitations"
            example_values = "comprehensive_motor_vehicle,windscreen_replacement,third_party_liability"
        elif "medical" in expert_lower or "health" in expert_lower:
            doc_type = "medical/healthcare document"
            customer_focus = "patients or healthcare professionals"
            example_keys = "diagnosis,treatment,medication,procedures,symptoms,contraindications"
            example_values = "acute_conditions,chronic_disease_management,dosage_guidelines"
        elif "math" in expert_lower or "scientific" in expert_lower:
            doc_type = "mathematical/scientific document"
            customer_focus = "students or researchers"
            example_keys = "theorems,formulas,proofs,definitions,applications,examples"
            example_values = "pythagorean_theorem,quadratic_equations,statistical_methods"
        elif "legal" in expert_lower:
            doc_type = "legal document"
            customer_focus = "clients seeking legal guidance"
            example_keys = "rights,obligations,definitions,procedures,requirements,remedies"
            example_values = "statutory_rights,contractual_obligations,filing_procedures"
        elif "technical" in expert_lower or "software" in expert_lower or "engineer" in expert_lower:
            doc_type = "technical documentation"
            customer_focus = "developers or technical users"
            example_keys = "features,configuration,api_methods,requirements,troubleshooting,security"
            example_values = "authentication_methods,data_structures,error_handling"
        else:
            # Generic fallback
            doc_type = "document"
            customer_focus = "readers"
            example_keys = "main_topics,concepts,procedures,definitions,guidelines,requirements"
            example_values = "specific_subtopics,detailed_aspects,particular_cases"
        
        prompt = f"""You are a {self.expert_type} analyzing this {doc_type}.

Read this COMPLETE document carefully:

{self.full_text}

{previous_context}

Task: Extract 6-10 key topics (keys) that {customer_focus} would ask about, with 5-8 specific sub-topics (values) each.

For each key-value pair, find ALL relevant citations in the document:
- Note the page number(s)
- Extract the section name
- Provide a short quote (60-80 chars)
- Provide a longer citation (up to 250 chars) for context

Requirements:
- Be SPECIFIC - reflect actual details in the document
- Each key-value must be UNIQUE (no duplicates)
- Stack multiple citations for the same key-value
- Cover different parts of the document
- Use snake_case

Example keys (adapt to this document): {example_keys}
Example values (adapt to this document): {example_values}

Return CSV format with these columns:
key,value,page,section,quote,long_citation,score,pass_id

Score: Your confidence (0.0-1.0) this citation supports the key-value
Pass_id: 1 if score >= 0.75, else 0

If same key-value has multiple citations, create ONE row with citations separated by " || "

Example output:
{example_keys.split(',')[0]},{example_values.split(',')[0]},2,Section Name,Short quote here...,Longer citation providing full context here...,0.95,1

Output the CSV data directly (no markdown, no explanations)."""

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=16000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        csv_content = response.content[0].text.strip()
        
        # Remove markdown if present
        if "```" in csv_content:
            csv_content = csv_content.split("```")[1]
            if csv_content.startswith("csv"):
                csv_content = csv_content[3:]
            csv_content = csv_content.strip()
        
        print(f"Received {len(csv_content.split(chr(10)))} rows from Claude")
        return csv_content
    
    def parse_csv_response(self, csv_content: str) -> List[Dict]:
        """Parse Claude's CSV response into structured data"""
        rows = []
        lines = csv_content.split('\n')
        
        # Skip header if present
        start_idx = 1 if lines[0].startswith('key,') else 0
        
        for line in lines[start_idx:]:
            line = line.strip()
            if not line:
                continue
            
            # Parse CSV line (handle commas in quotes)
            parts = []
            current = ""
            in_quotes = False
            
            for char in line:
                if char == '"':
                    in_quotes = not in_quotes
                elif char == ',' and not in_quotes:
                    parts.append(current)
                    current = ""
                    continue
                current += char
            parts.append(current)
            
            if len(parts) >= 8:
                try:
                    rows.append({
                        'key': parts[0].strip('"'),
                        'value': parts[1].strip('"'),
                        'page': parts[2].strip('"'),
                        'section': parts[3].strip('"'),
                        'quote': parts[4].strip('"'),
                        'long_citation': parts[5].strip('"'),
                        'score': float(parts[6].strip('"')),
                        'pass_id': int(parts[7].strip('"'))
                    })
                except (ValueError, IndexError) as e:
                    print(f"Warning: Skipping malformed row: {line[:50]}...")
                    continue
        
        return rows
    
    def normalize_key_value(self, text: str) -> str:
        """Normalize keys/values for deduplication"""
        # Remove trailing 's' for plurals
        normalized = text.lower().strip()
        if normalized.endswith('s') and len(normalized) > 3:
            singular = normalized[:-1]
            # Don't remove 's' if it would create a weird word
            if not singular.endswith('s'):
                normalized = singular
        
        # Normalize common synonyms
        synonyms = {
            'car': 'vehicle',
            'autos': 'vehicle',
            'automobile': 'vehicle',
            'timeframe': 'period',
            'time_frame': 'period',
        }
        
        for old, new in synonyms.items():
            normalized = normalized.replace(old, new)
        
        return normalized
    
    def get_semantic_clusters(self, items: List[str]) -> Dict[str, List[str]]:
        """Group semantically similar items using simple heuristics"""
        from collections import defaultdict
        
        # Extract root words (remove common suffixes)
        def get_root(word):
            # Remove common suffixes
            for suffix in ['_types', '_methods', '_process', '_services', '_management', 
                          '_administration', '_obligations', '_conditions', '_rights',
                          '_waiver', '_updates', '_changes', 's', 'es']:
                if word.endswith(suffix):
                    return word[:-len(suffix)]
            return word
        
        # Group by root words
        clusters = defaultdict(list)
        for item in items:
            root = get_root(item)
            clusters[root].append(item)
        
        # Merge clusters that share significant overlap
        merged = {}
        for root, items_list in clusters.items():
            # Find best representative (shortest, most common form)
            representative = min(items_list, key=lambda x: (len(x), x))
            merged[representative] = items_list
        
        return merged
    
    def deduplicate_citations(self, citations: List[Dict]) -> List[Dict]:
        """Merge semantically similar key-value pairs"""
        print("\nDeduplicating similar key-value pairs...")
        
        # Group citations by key first
        by_key = {}
        for citation in citations:
            key = citation['key']
            if key not in by_key:
                by_key[key] = []
            by_key[key].append(citation)
        
        # Cluster similar keys
        all_keys = list(by_key.keys())
        key_clusters = self.get_semantic_clusters(all_keys)
        
        print(f"\nKey consolidation:")
        for representative, similar_keys in key_clusters.items():
            if len(similar_keys) > 1:
                print(f"  {representative}: merging {similar_keys}")
        
        # Build new citation list with consolidated keys
        consolidated = []
        processed_keys = set()
        
        for representative, similar_keys in key_clusters.items():
            # Collect all citations for these similar keys
            key_citations = []
            for key in similar_keys:
                if key in by_key:
                    key_citations.extend(by_key[key])
                    processed_keys.add(key)
            
            # Now deduplicate values within this key
            value_groups = {}
            for citation in key_citations:
                value_norm = self.normalize_key_value(citation['value'])
                if value_norm not in value_groups:
                    value_groups[value_norm] = []
                value_groups[value_norm].append(citation)
            
            # Merge similar values
            for value_norm, group in value_groups.items():
                if len(group) == 1:
                    # Update key to representative
                    group[0]['key'] = representative
                    consolidated.append(group[0])
                else:
                    # Merge multiple citations
                    values = [c['value'] for c in group]
                    merged_value = min(values, key=lambda x: (len(x), x))  # Shortest form
                    
                    # Stack citations
                    pages = []
                    sections = []
                    quotes = []
                    long_citations = []
                    scores = []
                    
                    for c in group:
                        sections_parts = c['section'].split(' || ')
                        quotes_parts = c['quote'].split(' || ')
                        longs_parts = c['long_citation'].split(' || ')
                        
                        pages.append(str(c['page']))
                        sections.extend(sections_parts)
                        quotes.extend(quotes_parts)
                        long_citations.extend(longs_parts)
                        scores.append(c['score'])
                    
                    # Remove duplicates
                    seen = set()
                    unique_sections = []
                    unique_quotes = []
                    unique_longs = []
                    
                    for i, long in enumerate(long_citations):
                        if long not in seen:
                            seen.add(long)
                            if i < len(sections):
                                unique_sections.append(sections[i])
                            if i < len(quotes):
                                unique_quotes.append(quotes[i])
                            unique_longs.append(long)
                    
                    consolidated.append({
                        'key': representative,
                        'value': merged_value,
                        'page': pages[0],
                        'section': ' || '.join(unique_sections[:5]),
                        'quote': ' || '.join(unique_quotes[:5]),
                        'long_citation': ' || '.join(unique_longs[:5]),
                        'score': round(sum(scores) / len(scores), 4),
                        'pass_id': 1 if (sum(scores) / len(scores)) >= self.pass_threshold else 0
                    })
        
        print(f"\nReduced from {len(citations)} to {len(consolidated)} unique key-value pairs")
        print(f"Keys: {len(all_keys)} → {len(key_clusters)}")
        return consolidated
    
    def calculate_coverage(self, citations: List[Dict]) -> Dict:
        """Calculate document coverage"""
        pdf_words = set(re.findall(r'\b[a-z]{3,}\b', self.full_text.lower()))
        
        # Get all words from long_citations (split stacked citations)
        all_citation_text = []
        for c in citations:
            # Split by || for stacked citations
            citation_parts = c['long_citation'].split(' || ')
            all_citation_text.extend(citation_parts)
        
        citation_text = ' '.join(all_citation_text)
        citation_words = set(re.findall(r'\b[a-z]{3,}\b', citation_text.lower()))
        
        overlap = citation_words.intersection(pdf_words)
        coverage = len(overlap) / len(pdf_words) if pdf_words else 0
        
        return {
            'total_pdf_words': len(pdf_words),
            'total_citation_words': len(citation_words),
            'overlap_words': len(overlap),
            'coverage_percent': round(coverage * 100, 2)
        }
        """Calculate document coverage"""
        pdf_words = set(re.findall(r'\b[a-z]{3,}\b', self.full_text.lower()))
        
        # Get all words from long_citations (split stacked citations)
        all_citation_text = []
        for c in citations:
            # Split by || for stacked citations
            citation_parts = c['long_citation'].split(' || ')
            all_citation_text.extend(citation_parts)
        
        citation_text = ' '.join(all_citation_text)
        citation_words = set(re.findall(r'\b[a-z]{3,}\b', citation_text.lower()))
        
        overlap = citation_words.intersection(pdf_words)
        coverage = len(overlap) / len(pdf_words) if pdf_words else 0
        
        return {
            'total_pdf_words': len(pdf_words),
            'total_citation_words': len(citation_words),
            'overlap_words': len(overlap),
            'coverage_percent': round(coverage * 100, 2)
        }
    
    def process(self, output_csv: str, target_coverage: float = 60.0, max_iterations: int = 8):
        """
        Process document with multiple iterations until target coverage reached
        """
        print(f"Processing {self.pdf_path}...")
        print(f"Expert perspective: {self.expert_type}")
        print(f"Target coverage: {target_coverage}%")
        print(f"Max iterations: {max_iterations}\n")
        
        self.extract_text()
        
        all_citations = []
        processed_kvs = set()
        iteration = 1
        
        while iteration <= max_iterations:
            print(f"\n{'='*70}")
            print(f"ITERATION {iteration}/{max_iterations}")
            print(f"{'='*70}")
            
            # Get previous tags for context
            previous_tags = [f"{c['key']}:{c['value']}" for c in all_citations]
            
            # Analyze document
            csv_content = self.analyze_document_as_expert(iteration, previous_tags)
            
            # Parse response
            new_citations = self.parse_csv_response(csv_content)
            
            # Filter out duplicates
            unique_new = []
            for citation in new_citations:
                kv = f"{citation['key']}:{citation['value']}"
                if kv not in processed_kvs:
                    processed_kvs.add(kv)
                    unique_new.append(citation)
                else:
                    print(f"  Skipping duplicate: {kv}")
            
            all_citations.extend(unique_new)
            
            # Calculate coverage (without deduplication yet)
            coverage = self.calculate_coverage(all_citations)
            
            print(f"\nIteration {iteration} Results:")
            print(f"  New unique key-values: {len(unique_new)}")
            print(f"  Total unique key-values: {len(all_citations)}")
            print(f"  Unique keys: {len(set(c['key'] for c in all_citations))}")
            print(f"  Average score: {sum(c['score'] for c in all_citations) / len(all_citations):.3f}")
            print(f"  Coverage: {coverage['coverage_percent']}%")
            
            # Check if target reached
            if coverage['coverage_percent'] >= target_coverage:
                print(f"\n✓ Target coverage of {target_coverage}% reached!")
                break
            
            iteration += 1
        
        # Deduplicate similar key-values once at the end
        all_citations = self.deduplicate_citations(all_citations)
        
        # Recalculate final coverage after deduplication
        coverage = self.calculate_coverage(all_citations)
        
        # Write final results
        print(f"\nWriting {len(all_citations)} key-value pairs to {output_csv}...")
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['key', 'value', 'page', 'section', 'quote', 'long_citation', 'score', 'pass_id'])
            writer.writeheader()
            writer.writerows(all_citations)
        
        # Final summary
        print("\n" + "="*70)
        print("FINAL SUMMARY")
        print("="*70)
        print(f"Total unique key-value pairs: {len(all_citations)}")
        print(f"Unique keys: {len(set(c['key'] for c in all_citations))}")
        print(f"Average score: {sum(c['score'] for c in all_citations) / len(all_citations):.3f}")
        print(f"Pass rate: {sum(c['pass_id'] for c in all_citations) / len(all_citations) * 100:.1f}%")
        print(f"\nDocument Coverage:")
        print(f"  Total unique words in PDF: {coverage['total_pdf_words']}")
        print(f"  Unique words in citations: {coverage['total_citation_words']}")
        print(f"  Word overlap: {coverage['overlap_words']}")
        print(f"  Coverage: {coverage['coverage_percent']}%")
        
        # Breakdown by key
        print(f"\nCoverage by Topic:")
        key_counts = Counter([c['key'] for c in all_citations])
        for key, count in sorted(key_counts.items()):
            print(f"  {key}: {count} sub-topics")
        print("="*70)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python pdf_tagger.py input.pdf output.csv [expert_type] [target_coverage] [max_iterations]")
        print("\nExpert types:")
        print("  'insurance expert' - for insurance policies (default)")
        print("  'medical professional' - for healthcare documents")
        print("  'mathematics professor' - for math/science papers")
        print("  'legal expert' - for legal documents")
        print("  'software engineer' - for technical documentation")
        print("  Or any custom expert description")
        print("\nExamples:")
        print("  python pdf_tagger.py policy.pdf output.csv")
        print("  python pdf_tagger.py policy.pdf output.csv 'insurance expert' 60 2")
        print("  python pdf_tagger.py medical.pdf output.csv 'medical professional' 70")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    output_csv = sys.argv[2]
    expert_type = sys.argv[3] if len(sys.argv) > 3 else "insurance expert"
    target_coverage = float(sys.argv[4]) if len(sys.argv) > 4 else 60.0
    max_iterations = int(sys.argv[5]) if len(sys.argv) > 5 else 8
    
    print(f"Expert perspective: {expert_type}")
    print(f"Target coverage: {target_coverage}%")
    print(f"Max iterations: {max_iterations}\n")
    
    tagger = PDFTagger(pdf_path, expert_type=expert_type)
    tagger.process(output_csv, target_coverage=target_coverage, max_iterations=max_iterations)