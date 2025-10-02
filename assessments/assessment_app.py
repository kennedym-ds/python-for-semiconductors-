"""
Streamlit Assessment Application for Python for Semiconductors

Interactive web application for taking assessments, tracking progress,
and visualizing learning outcomes.

Usage:
    streamlit run assessments/assessment_app.py
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ============================================================================
# Configuration
# ============================================================================

ASSESSMENTS_DIR = Path(__file__).parent
DB_PATH = ASSESSMENTS_DIR / "assessment_results.db"

# Available modules
AVAILABLE_MODULES = {
    "module-1": "Module 1: Python & Data Fundamentals",
    "module-2": "Module 2: Data Quality & Statistical Analysis",
    "module-3": "Module 3: Introduction to Machine Learning",
    "module-4": "Module 4: Advanced ML Techniques",
    "module-5": "Module 5: Ensemble Methods & Time Series",
    "module-6": "Module 6: Deep Learning Fundamentals",
    "module-7": "Module 7: Computer Vision for Defect Detection",
    "module-8": "Module 8: Generative AI for Semiconductors",
    "module-9": "Module 9: MLOps & Deployment",
    "module-10": "Module 10: Capstone Projects",
    "module-11": "Module 11: Edge AI & Model Optimization",
}


# ============================================================================
# Database Functions
# ============================================================================


def init_database():
    """Initialize SQLite database for storing results."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Create users table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            username TEXT NOT NULL,
            email TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """
    )

    # Create assessment_attempts table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS assessment_attempts (
            attempt_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            module_id TEXT NOT NULL,
            sub_module TEXT NOT NULL,
            started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP,
            score REAL,
            max_score REAL,
            percentage REAL,
            time_taken_seconds INTEGER,
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        )
    """
    )

    # Create question_responses table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS question_responses (
            response_id INTEGER PRIMARY KEY AUTOINCREMENT,
            attempt_id INTEGER NOT NULL,
            question_id TEXT NOT NULL,
            question_type TEXT NOT NULL,
            user_answer TEXT,
            correct_answer TEXT,
            is_correct BOOLEAN,
            points_earned REAL,
            points_possible REAL,
            FOREIGN KEY (attempt_id) REFERENCES assessment_attempts(attempt_id)
        )
    """
    )

    conn.commit()
    conn.close()


def create_or_get_user(user_id: str, username: str, email: Optional[str] = None) -> bool:
    """Create new user or retrieve existing user."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        cursor.execute(
            """
            INSERT OR IGNORE INTO users (user_id, username, email)
            VALUES (?, ?, ?)
        """,
            (user_id, username, email),
        )
        conn.commit()
        return True
    except Exception as e:
        st.error(f"Error creating user: {e}")
        return False
    finally:
        conn.close()


def start_assessment_attempt(user_id: str, module_id: str, sub_module: str) -> int:
    """Start a new assessment attempt and return attempt_id."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO assessment_attempts (user_id, module_id, sub_module)
        VALUES (?, ?, ?)
    """,
        (user_id, module_id, sub_module),
    )
    attempt_id = cursor.lastrowid
    conn.commit()
    conn.close()

    return attempt_id


def complete_assessment_attempt(attempt_id: int, score: float, max_score: float, time_taken: int):
    """Mark assessment as completed with final score."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    percentage = (score / max_score * 100) if max_score > 0 else 0

    cursor.execute(
        """
        UPDATE assessment_attempts
        SET completed_at = CURRENT_TIMESTAMP,
            score = ?,
            max_score = ?,
            percentage = ?,
            time_taken_seconds = ?
        WHERE attempt_id = ?
    """,
        (score, max_score, percentage, time_taken, attempt_id),
    )

    conn.commit()
    conn.close()


def save_question_response(
    attempt_id: int,
    question_id: str,
    question_type: str,
    user_answer: str,
    correct_answer: str,
    is_correct: bool,
    points_earned: float,
    points_possible: float,
):
    """Save individual question response."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO question_responses
        (attempt_id, question_id, question_type, user_answer, correct_answer,
         is_correct, points_earned, points_possible)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """,
        (
            attempt_id,
            question_id,
            question_type,
            user_answer,
            correct_answer,
            is_correct,
            points_earned,
            points_possible,
        ),
    )

    conn.commit()
    conn.close()


def get_user_progress(user_id: str) -> pd.DataFrame:
    """Get all assessment attempts for a user."""
    conn = sqlite3.connect(DB_PATH)

    query = """
        SELECT
            module_id,
            sub_module,
            completed_at,
            score,
            max_score,
            percentage,
            time_taken_seconds
        FROM assessment_attempts
        WHERE user_id = ? AND completed_at IS NOT NULL
        ORDER BY completed_at DESC
    """

    df = pd.read_sql_query(query, conn, params=(user_id,))
    conn.close()

    return df


def get_user_stats(user_id: str) -> Dict:
    """Get summary statistics for a user."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Total assessments completed
    cursor.execute(
        """
        SELECT COUNT(*) FROM assessment_attempts
        WHERE user_id = ? AND completed_at IS NOT NULL
    """,
        (user_id,),
    )
    total_completed = cursor.fetchone()[0]

    # Average score
    cursor.execute(
        """
        SELECT AVG(percentage) FROM assessment_attempts
        WHERE user_id = ? AND completed_at IS NOT NULL
    """,
        (user_id,),
    )
    avg_score = cursor.fetchone()[0] or 0

    # Total time spent (in hours)
    cursor.execute(
        """
        SELECT SUM(time_taken_seconds) FROM assessment_attempts
        WHERE user_id = ? AND completed_at IS NOT NULL
    """,
        (user_id,),
    )
    total_seconds = cursor.fetchone()[0] or 0
    total_hours = total_seconds / 3600

    # Modules completed
    cursor.execute(
        """
        SELECT COUNT(DISTINCT module_id) FROM assessment_attempts
        WHERE user_id = ? AND completed_at IS NOT NULL
    """,
        (user_id,),
    )
    modules_completed = cursor.fetchone()[0]

    conn.close()

    return {
        "total_completed": total_completed,
        "avg_score": round(avg_score, 1),
        "total_hours": round(total_hours, 2),
        "modules_completed": modules_completed,
    }


# ============================================================================
# Question Loading
# ============================================================================


def load_questions(module_id: str, sub_module: str) -> Optional[Dict]:
    """Load questions from JSON file."""
    module_dir = ASSESSMENTS_DIR / module_id
    question_file = module_dir / f"{sub_module}-questions.json"

    if not question_file.exists():
        return None

    try:
        with open(question_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading questions: {e}")
        return None


def get_available_sub_modules(module_id: str) -> List[str]:
    """Get list of available sub-modules for a module."""
    module_dir = ASSESSMENTS_DIR / module_id

    if not module_dir.exists():
        return []

    sub_modules = []
    for file in module_dir.glob("*-questions.json"):
        # Extract sub-module ID (e.g., "1.1" from "1.1-questions.json")
        sub_module = file.stem.rsplit("-questions", 1)[0]
        sub_modules.append(sub_module)

    return sorted(sub_modules)


# ============================================================================
# Question Rendering
# ============================================================================


def render_multiple_choice(question: Dict, question_idx: int) -> Optional[int]:
    """Render multiple choice question and return selected index."""
    st.markdown(f"**Question {question_idx + 1}** ({question['points']} points)")
    st.markdown(f"**Difficulty:** {question['difficulty'].title()}")
    st.markdown(question["question"])

    options = question.get("options", [])
    if not options:
        st.error("No options available for this question")
        return None

    # Use radio buttons for single selection
    selected = st.radio(
        "Select your answer:",
        options=range(len(options)),
        format_func=lambda x: options[x],
        key=f"mc_{question['id']}",
    )

    return selected


def render_coding_exercise(question: Dict, question_idx: int) -> str:
    """Render coding exercise question and return code answer."""
    st.markdown(f"**Question {question_idx + 1}** ({question['points']} points)")
    st.markdown(f"**Difficulty:** {question['difficulty'].title()}")
    st.markdown(question["question"])

    # Show starter code if available
    starter_code = question.get("starter_code", "")
    if starter_code:
        st.markdown("**Starter Code:**")
        st.code(starter_code, language="python")

    # Code editor
    user_code = st.text_area(
        "Enter your code:",
        value=starter_code,
        height=200,
        key=f"code_{question['id']}",
    )

    return user_code


def render_conceptual(question: Dict, question_idx: int) -> str:
    """Render conceptual question and return text answer."""
    st.markdown(f"**Question {question_idx + 1}** ({question['points']} points)")
    st.markdown(f"**Difficulty:** {question['difficulty'].title()}")
    st.markdown(question["question"])

    # Show rubric if available
    rubric = question.get("rubric", [])
    if rubric:
        st.markdown("**Grading Criteria:**")
        for criterion in rubric:
            st.markdown(f"- {criterion.get('criteria', 'N/A')} ({criterion.get('points', 0)} points)")

    # Text area for answer
    user_answer = st.text_area(
        "Enter your answer:",
        height=150,
        key=f"conceptual_{question['id']}",
    )

    return user_answer


# ============================================================================
# Grading Functions
# ============================================================================


def grade_multiple_choice(user_answer: int, correct_answer: int, points: int) -> Tuple[bool, float]:
    """Grade multiple choice question."""
    is_correct = user_answer == correct_answer
    points_earned = points if is_correct else 0
    return is_correct, points_earned


def grade_coding_exercise(user_code: str, test_cases: List[Dict], points: int) -> Tuple[bool, float]:
    """
    Grade coding exercise (simplified - in production, would use sandboxed execution).
    For now, just check if code is not empty and award partial credit.
    """
    if not user_code or user_code.strip() == "":
        return False, 0

    # In production, would execute test cases in sandbox
    # For now, award partial credit if code is provided
    return True, points * 0.8  # 80% credit for attempting


def grade_conceptual(user_answer: str, rubric: List[Dict], points: int) -> Tuple[bool, float]:
    """
    Grade conceptual question (simplified - requires manual grading in production).
    For now, check for keywords and award partial credit.
    """
    if not user_answer or user_answer.strip() == "":
        return False, 0

    # Check for keywords from rubric
    points_earned = 0
    for criterion in rubric:
        keywords = criterion.get("keywords", [])
        if any(keyword.lower() in user_answer.lower() for keyword in keywords):
            points_earned += criterion.get("points", 0)

    # Cap at maximum points
    points_earned = min(points_earned, points)
    is_correct = points_earned >= points * 0.7  # 70% threshold

    return is_correct, points_earned


# ============================================================================
# Streamlit App Pages
# ============================================================================


def page_login():
    """Login/Registration page."""
    st.title("🎓 Python for Semiconductors Assessment System")

    st.markdown(
        """
    Welcome to the interactive assessment system! Track your progress through
    the learning modules and visualize your improvement over time.
    """
    )

    with st.form("login_form"):
        st.subheader("Login or Register")

        user_id = st.text_input("User ID", help="Enter a unique user ID")
        username = st.text_input("Display Name")
        email = st.text_input("Email (optional)")

        submit = st.form_submit_button("Continue")

        if submit:
            if not user_id or not username:
                st.error("Please enter both User ID and Display Name")
            else:
                if create_or_get_user(user_id, username, email):
                    st.session_state.user_id = user_id
                    st.session_state.username = username
                    st.rerun()


def page_select_assessment():
    """Assessment selection page."""
    st.title(f"Welcome, {st.session_state.username}! 👋")

    # Show quick stats
    stats = get_user_stats(st.session_state.user_id)
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Assessments Completed", stats["total_completed"])
    with col2:
        st.metric("Average Score", f"{stats['avg_score']}%")
    with col3:
        st.metric("Study Time", f"{stats['total_hours']}h")
    with col4:
        st.metric("Modules Completed", stats["modules_completed"])

    st.divider()

    st.subheader("Select Assessment")

    # Module selection
    selected_module = st.selectbox(
        "Choose Module",
        options=list(AVAILABLE_MODULES.keys()),
        format_func=lambda x: AVAILABLE_MODULES[x],
    )

    # Sub-module selection
    sub_modules = get_available_sub_modules(selected_module)

    if not sub_modules:
        st.warning(f"No assessments available for {AVAILABLE_MODULES[selected_module]}")
        return

    selected_sub_module = st.selectbox("Choose Sub-Module", options=sub_modules, format_func=lambda x: x.title())

    # Load and show question count
    questions_data = load_questions(selected_module, selected_sub_module)

    if questions_data:
        num_questions = len(questions_data.get("questions", []))
        total_points = sum(q.get("points", 1) for q in questions_data.get("questions", []))

        st.info(f"📝 This assessment has **{num_questions} questions** worth **{total_points} points** total")

        if st.button("Start Assessment", type="primary"):
            st.session_state.current_module = selected_module
            st.session_state.current_sub_module = selected_sub_module
            st.session_state.questions_data = questions_data
            st.session_state.current_question = 0
            st.session_state.answers = {}
            st.session_state.start_time = datetime.now()
            st.session_state.page = "take_assessment"
            st.rerun()


def page_take_assessment():
    """Assessment taking page."""
    questions = st.session_state.questions_data.get("questions", [])
    current_idx = st.session_state.current_question

    if current_idx >= len(questions):
        # Assessment complete
        page_results()
        return

    question = questions[current_idx]

    # Progress bar
    progress = (current_idx + 1) / len(questions)
    st.progress(progress, text=f"Question {current_idx + 1} of {len(questions)}")

    st.title(st.session_state.questions_data.get("title", "Assessment"))

    # Render question based on type
    q_type = question.get("type")

    if q_type == "multiple_choice":
        answer = render_multiple_choice(question, current_idx)
    elif q_type == "coding_exercise":
        answer = render_coding_exercise(question, current_idx)
    elif q_type == "conceptual":
        answer = render_conceptual(question, current_idx)
    else:
        st.error(f"Unknown question type: {q_type}")
        answer = None

    st.divider()

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        if current_idx > 0:
            if st.button("← Previous"):
                st.session_state.current_question -= 1
                st.rerun()

    with col2:
        if st.button("Save Answer"):
            if answer is not None:
                st.session_state.answers[question["id"]] = answer
                st.success("Answer saved!")
            else:
                st.warning("Please select or enter an answer")

    with col3:
        if current_idx < len(questions) - 1:
            if st.button("Next →"):
                if answer is not None:
                    st.session_state.answers[question["id"]] = answer
                st.session_state.current_question += 1
                st.rerun()
        else:
            if st.button("Submit Assessment", type="primary"):
                if answer is not None:
                    st.session_state.answers[question["id"]] = answer
                st.session_state.current_question += 1
                st.rerun()


def page_results():
    """Results and grading page."""
    st.title("📊 Assessment Results")

    questions = st.session_state.questions_data.get("questions", [])
    answers = st.session_state.answers

    # Calculate time taken
    time_taken = (datetime.now() - st.session_state.start_time).seconds

    # Start database attempt
    attempt_id = start_assessment_attempt(
        st.session_state.user_id,
        st.session_state.current_module,
        st.session_state.current_sub_module,
    )

    # Grade each question
    total_score = 0
    max_score = 0
    results = []

    for question in questions:
        q_id = question["id"]
        q_type = question["type"]
        points = question.get("points", 1)
        max_score += points

        user_answer = answers.get(q_id)

        if user_answer is None:
            # No answer provided
            is_correct = False
            points_earned = 0
            correct_answer = "N/A"
        elif q_type == "multiple_choice":
            correct_idx = question.get("correct_answer")
            is_correct, points_earned = grade_multiple_choice(user_answer, correct_idx, points)
            correct_answer = str(correct_idx)
        elif q_type == "coding_exercise":
            test_cases = question.get("test_cases", [])
            is_correct, points_earned = grade_coding_exercise(user_answer, test_cases, points)
            correct_answer = "See test cases"
        elif q_type == "conceptual":
            rubric = question.get("rubric", [])
            is_correct, points_earned = grade_conceptual(user_answer, rubric, points)
            correct_answer = "See rubric"
        else:
            is_correct = False
            points_earned = 0
            correct_answer = "Unknown"

        total_score += points_earned

        results.append(
            {
                "question_id": q_id,
                "question": question.get("question", "")[:50] + "...",
                "type": q_type,
                "is_correct": is_correct,
                "points_earned": points_earned,
                "points_possible": points,
            }
        )

        # Save to database
        save_question_response(
            attempt_id,
            q_id,
            q_type,
            str(user_answer),
            str(correct_answer),
            is_correct,
            points_earned,
            points,
        )

    # Complete assessment
    percentage = (total_score / max_score * 100) if max_score > 0 else 0
    complete_assessment_attempt(attempt_id, total_score, max_score, time_taken)

    # Display results
    st.success(f"Assessment completed in {time_taken // 60} minutes {time_taken % 60} seconds")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Score", f"{total_score}/{max_score}")
    with col2:
        st.metric("Percentage", f"{percentage:.1f}%")
    with col3:
        status = "✅ PASSED" if percentage >= 70 else "❌ NEEDS IMPROVEMENT"
        st.metric("Status", status)

    # Results table
    st.subheader("Question Breakdown")
    results_df = pd.DataFrame(results)
    st.dataframe(results_df, use_container_width=True)

    # Feedback
    st.subheader("Feedback")
    if percentage >= 90:
        st.success("🎉 Outstanding work! You've mastered the concepts.")
    elif percentage >= 80:
        st.success("👍 Great job! You're ready to move forward.")
    elif percentage >= 70:
        st.info("✓ Good work! Review areas where you lost points.")
    else:
        st.warning("📚 Additional study recommended. Focus on core concepts.")

    if st.button("Back to Dashboard"):
        # Clear assessment state
        for key in [
            "current_module",
            "current_sub_module",
            "questions_data",
            "current_question",
            "answers",
            "start_time",
        ]:
            if key in st.session_state:
                del st.session_state[key]
        st.session_state.page = "select"
        st.rerun()


def page_progress():
    """Progress tracking and visualization page."""
    st.title("📈 Your Progress")

    user_id = st.session_state.user_id

    # Get progress data
    progress_df = get_user_progress(user_id)

    if progress_df.empty:
        st.info("No assessments completed yet. Start your first assessment!")
        return

    # Summary stats
    stats = get_user_stats(user_id)
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Assessments", stats["total_completed"])
    with col2:
        st.metric("Average Score", f"{stats['avg_score']}%")
    with col3:
        st.metric("Study Time", f"{stats['total_hours']}h")
    with col4:
        st.metric("Modules Completed", stats["modules_completed"])

    st.divider()

    # Progress over time
    st.subheader("Score Progression")

    progress_df["completed_at"] = pd.to_datetime(progress_df["completed_at"])
    progress_df = progress_df.sort_values("completed_at")

    fig_line = px.line(
        progress_df,
        x="completed_at",
        y="percentage",
        title="Assessment Scores Over Time",
        labels={"completed_at": "Date", "percentage": "Score (%)"},
        markers=True,
    )
    fig_line.add_hline(y=70, line_dash="dash", line_color="orange", annotation_text="Passing (70%)")
    fig_line.add_hline(y=90, line_dash="dash", line_color="green", annotation_text="Excellent (90%)")
    st.plotly_chart(fig_line, use_container_width=True)

    # Scores by module
    st.subheader("Performance by Module")

    module_stats = (
        progress_df.groupby("module_id")
        .agg({"percentage": "mean", "module_id": "count"})
        .rename(columns={"module_id": "attempts"})
        .reset_index()
    )

    fig_bar = px.bar(
        module_stats,
        x="module_id",
        y="percentage",
        title="Average Score by Module",
        labels={"module_id": "Module", "percentage": "Average Score (%)"},
        text="percentage",
    )
    fig_bar.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    st.plotly_chart(fig_bar, use_container_width=True)

    # Recent assessments
    st.subheader("Recent Assessments")
    recent = progress_df.head(10)[["completed_at", "module_id", "sub_module", "percentage", "time_taken_seconds"]]
    recent["time_taken"] = recent["time_taken_seconds"].apply(lambda x: f"{x // 60}m {x % 60}s")
    recent = recent.drop(columns=["time_taken_seconds"])
    st.dataframe(recent, use_container_width=True)


# ============================================================================
# Main App
# ============================================================================


def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="Python for Semiconductors Assessments",
        page_icon="🎓",
        layout="wide",
    )

    # Initialize database
    init_database()

    # Initialize session state
    if "user_id" not in st.session_state:
        st.session_state.user_id = None
    if "page" not in st.session_state:
        st.session_state.page = "select"

    # Sidebar navigation
    if st.session_state.user_id:
        with st.sidebar:
            st.title("Navigation")

            if st.button("🏠 Dashboard"):
                st.session_state.page = "select"
                st.rerun()

            if st.button("📈 Progress"):
                st.session_state.page = "progress"
                st.rerun()

            st.divider()

            st.markdown(f"**User:** {st.session_state.username}")
            st.markdown(f"**ID:** {st.session_state.user_id}")

            if st.button("Logout"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()

    # Route to appropriate page
    if not st.session_state.user_id:
        page_login()
    elif st.session_state.page == "select":
        page_select_assessment()
    elif st.session_state.page == "take_assessment":
        page_take_assessment()
    elif st.session_state.page == "progress":
        page_progress()


if __name__ == "__main__":
    main()
