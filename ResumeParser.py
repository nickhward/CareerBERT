import re
from pdfminer.high_level import extract_text


class ResumeParserClass:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.resume_text = self.extract_text_from_pdf()


    def extract_text_from_pdf(self):
        text = extract_text(self.pdf_path)
        return text

    def extract_section(self, section_header):
        try:
            section = re.search(f'{section_header}(.+?)(?=(EDUCATION|EXPERIENCE|PERSONALPROJECTS|PUBLICATIONS|TECHNICALSKILLS|$))', self.resume_text, re.DOTALL).group(1)
            return ' '.join(section.split())
        except AttributeError:
            return "Section not found."

    def parse(self):
        # Define headers for each section
        education_header = 'EDUCATION'
        experience_header = 'EXPERIENCE'
        projects_header = 'PERSONAL PROJECTS'
        publications_header = 'PUBLICATIONS'
        skills_header = 'TECHNICAL SKILLS'

        # Extract each section
        education = self.extract_section(education_header)
        experience = self.extract_section(experience_header)
        projects = self.extract_section(projects_header)
        publications = self.extract_section(publications_header)
        skills = self.extract_section(skills_header)

        return {
            "Education": education,
            "Experience": experience,
            "Projects": projects,
            "Publications": publications,
            "Skills": skills
        }



