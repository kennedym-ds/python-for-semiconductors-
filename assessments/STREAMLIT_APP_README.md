# Streamlit Assessment Application

Interactive web application for taking assessments, tracking progress, and visualizing learning outcomes in the Python for Semiconductors learning series.

## Features

### 🎓 Assessment Taking

- **Interactive Quiz Interface**: Take assessments with a clean, user-friendly interface
- **Multiple Question Types**:
  - Multiple Choice: Radio button selection
  - Coding Exercises: Code editor with syntax highlighting
  - Conceptual Questions: Text area for written responses
- **Progress Tracking**: Visual progress bar showing current question
- **Navigation**: Move back and forward between questions
- **Auto-Save**: Answers are saved as you progress

### 📊 Progress Visualization

- **Score Progression**: Line chart showing scores over time
- **Module Performance**: Bar charts comparing performance across modules
- **Summary Statistics**:
  - Total assessments completed
  - Average score across all attempts
  - Total study time
  - Modules completed
- **Recent Activity**: Table of recent assessment attempts

### 💾 Data Persistence

- **SQLite Database**: All results stored locally in `assessment_results.db`
- **User Profiles**: Track multiple users with unique IDs
- **Detailed Records**: Question-level response tracking
- **Historical Data**: Complete history of all assessment attempts

## Installation

### Step 1: Install Dependencies

```powershell
# Install Streamlit and visualization libraries
pip install -r requirements-streamlit.txt
```

### Step 2: Verify Installation

```powershell
streamlit --version
```

## Usage

### Starting the Application

From the repository root:

```powershell
streamlit run assessments/assessment_app.py
```

The app will open automatically in your default browser at `http://localhost:8501`

### First Time Setup

1. **Login/Register**:
   - Enter a unique User ID (e.g., `student_001`)
   - Enter your display name
   - Optionally add your email
   - Click "Continue"

2. **Dashboard**:
   - View your summary statistics
   - Select a module from the dropdown
   - Choose a sub-module (e.g., `1.1`, `2.1`)
   - Click "Start Assessment"

3. **Taking Assessment**:
   - Read each question carefully
   - Select/enter your answer
   - Click "Save Answer" to save progress
   - Use "Next" to move forward, "Previous" to go back
   - Click "Submit Assessment" on the last question

4. **View Results**:
   - See your score and percentage
   - Review question breakdown
   - Read personalized feedback
   - Return to dashboard

5. **Track Progress**:
   - Click "Progress" in the sidebar
   - View score trends over time
   - Compare performance across modules
   - See recent assessment history

## Application Structure

```
assessments/
├── assessment_app.py          # Main Streamlit application
├── assessment_results.db      # SQLite database (created on first run)
├── module-1/                  # Module 1 questions
│   ├── 1.1-questions.json
│   └── 1.2-questions.json
├── module-2/                  # Module 2 questions
│   └── ...
└── ...
```

## Database Schema

### Users Table

- `user_id` (Primary Key): Unique user identifier
- `username`: Display name
- `email`: Optional email address
- `created_at`: Registration timestamp

### Assessment Attempts Table

- `attempt_id` (Primary Key): Unique attempt identifier
- `user_id` (Foreign Key): User who took the assessment
- `module_id`: Module identifier (e.g., "module-1")
- `sub_module`: Sub-module identifier (e.g., "1.1")
- `started_at`: Start timestamp
- `completed_at`: Completion timestamp
- `score`: Points earned
- `max_score`: Total points possible
- `percentage`: Score percentage
- `time_taken_seconds`: Time taken to complete

### Question Responses Table

- `response_id` (Primary Key): Unique response identifier
- `attempt_id` (Foreign Key): Associated attempt
- `question_id`: Question identifier
- `question_type`: Type of question
- `user_answer`: User's answer
- `correct_answer`: Correct answer
- `is_correct`: Boolean correct/incorrect
- `points_earned`: Points earned for this question
- `points_possible`: Points possible for this question

## Grading Logic

### Multiple Choice

- **Correct**: Full points awarded
- **Incorrect**: Zero points

### Coding Exercises

- **Production Note**: In a production system, code would be executed in a sandboxed environment with test cases
- **Current Implementation**: Partial credit (80%) awarded if code is provided
- **Future Enhancement**: Integration with code execution engine

### Conceptual Questions

- **Keyword Matching**: Checks for keywords from rubric in answer
- **Partial Credit**: Points awarded based on matching criteria
- **Threshold**: 70% of points required to mark as "correct"
- **Production Note**: Requires manual grading or ML-based evaluation for accuracy

## Features by Page

### Login Page

- User registration and authentication
- Email capture (optional)
- Auto-login for returning users

### Dashboard (Select Assessment)

- Quick stats overview (4 metrics)
- Module selection dropdown
- Sub-module selection based on available questions
- Assessment preview (question count, total points)

### Assessment Taking Page

- Progress indicator (e.g., "Question 3 of 20")
- Question rendering based on type
- Answer input (radio buttons, code editor, or text area)
- Navigation controls (Previous, Save, Next/Submit)
- Auto-save on navigation

### Results Page

- Overall score display (3 metrics)
- Question-by-question breakdown table
- Personalized feedback based on performance
- Return to dashboard button

### Progress Page

- Summary statistics (4 metrics)
- Line chart: Score progression over time with passing thresholds
- Bar chart: Average score by module
- Recent assessments table (last 10)

## Configuration

### Passing Score

Default: 70% (defined in grading logic)

### Available Modules

All 11 modules are available:

- Module 1: Python & Data Fundamentals
- Module 2: Data Quality & Statistical Analysis
- Module 3: Introduction to Machine Learning
- Module 4: Advanced ML Techniques
- Module 5: Ensemble Methods & Time Series
- Module 6: Deep Learning Fundamentals
- Module 7: Computer Vision for Defect Detection
- Module 8: Generative AI for Semiconductors
- Module 9: MLOps & Deployment
- Module 10: Capstone Projects
- Module 11: Edge AI & Model Optimization

## Troubleshooting

### App Won't Start

```powershell
# Reinstall dependencies
pip install --upgrade streamlit plotly pandas

# Check Python version (requires 3.8+)
python --version
```

### Database Errors

```powershell
# Delete and recreate database
Remove-Item assessments/assessment_results.db
# Restart the app
```

### Questions Not Loading

- Verify question JSON files exist in module directories
- Check file naming: `X.Y-questions.json` format
- Validate JSON syntax using validation script:

  ```powershell
  python assessments/validation/validate_all.py --module 1
  ```

### Port Already in Use

```powershell
# Use a different port
streamlit run assessments/assessment_app.py --server.port 8502
```

## Development Notes

### Adding New Features

**Custom Grading Logic**:

- Modify `grade_*` functions in `assessment_app.py`
- Add new question types in rendering section

**New Visualizations**:

- Add Plotly charts in `page_progress()` function
- Query database for new metrics

**Export Functionality**:

- Add CSV/PDF export buttons
- Use pandas `.to_csv()` or ReportLab for PDF

### Testing

```powershell
# Test with sample user
# User ID: test_user_001
# Username: Test User

# Take multiple assessments to generate data
# View progress page to verify visualizations
```

## Security Considerations

**Current Implementation** (Development/Education):

- Local SQLite database (not suitable for production)
- No password authentication
- User IDs are self-assigned
- Code execution is simulated (not sandboxed)

**Production Enhancements Needed**:

- Authentication system (OAuth, JWT)
- Password hashing
- Sandboxed code execution (Docker, AWS Lambda)
- Cloud database (PostgreSQL, MongoDB)
- HTTPS/SSL
- Rate limiting
- Input sanitization

## Future Enhancements

- [ ] Timer for timed assessments
- [ ] Question randomization
- [ ] Attempt limits per assessment
- [ ] Peer comparison (anonymized leaderboards)
- [ ] Export results to PDF/CSV
- [ ] Email notifications for completion
- [ ] Mobile-responsive design improvements
- [ ] Dark mode theme
- [ ] Multi-language support
- [ ] Instructor dashboard for grading conceptual questions
- [ ] Sandboxed code execution for coding exercises
- [ ] AI-powered feedback using LLMs
- [ ] Adaptive difficulty based on performance

## Support

For issues or questions:

1. Check the troubleshooting section above
2. Validate question files with the validation script
3. Review database schema for data integrity
4. Check Streamlit logs in the terminal

## License

Part of the Python for Semiconductors learning series. See main repository README for license information.
