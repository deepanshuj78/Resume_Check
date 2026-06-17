import spacy
TECH_SKILLS_BANK = {
    "languages": ["python", "java", "sql", "javascript", "typescript", "html", "css", "c++", "ruby", "go"],
    "frameworks": ["react", "node.js", "node", "angular", "vue", "fastapi", "flask", "django", "express"],
    "data_science": ["machine learning", "data science", "nlp", "pandas", "numpy", "pytorch", "tensorflow", "scikit-learn"],
    "cloud_devops": ["aws", "azure", "docker", "kubernetes", "git", "ci/cd", "jenkins"]
}
FLAT_SKILLS_BANK = [skill for sublist in TECH_SKILLS_BANK.values() for skill in sublist]
DOMAIN_TRAINING_DATA = [
    "python data science machine learning model analytics sql pandas numpy pytorch tensorflow",
    "recruitment payroll onboarding human resources interview hiring employee benefits talent management",
    "javascript react html css node frontend developer web app java design typescript angular",
    "aws azure docker kubernetes cicd jenkins linux devops infrastructure automation git"
]
DOMAIN_LABELS = ["Data Scientist", "HR Specialist", "Web Developer", "DevOps Engineer"]
SPACY_STOPS = set(spacy.lang.en.stop_words.STOP_WORDS)
CUSTOM_EXTENSIONS = {"company", "job", "person", "team", "role", "work", "responsibilities", "resume", "experience"}
FINAL_STOP_WORDS = list(SPACY_STOPS.union(CUSTOM_EXTENSIONS))