#!/usr/bin/env python3
"""
Educational Assessment System for Python for Semiconductors Learning Series

This module provides comprehensive assessment capabilities for validating
learning outcomes and tracking student progress through the curriculum.

Usage Examples:
    # Run module assessment
    python assessment_system.py assess --module module-3 --type knowledge
    
    # Generate progress report
    python assessment_system.py progress --student_id user123 --format json
    
    # Create assessment for new module
    python assessment_system.py create --module module-5 --questions questions.json
"""

import argparse
import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import sys
import os

# Add modules path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

RANDOM_SEED = 42

@dataclass
class Question:
    """Individual assessment question."""
    id: str
    type: str  # multiple_choice, coding, conceptual
    question: str
    options: Optional[List[str]] = None
    correct_answer: Optional[str] = None
    points: int = 1
    difficulty: str = "medium"  # easy, medium, hard
    topic: str = ""

@dataclass 
class AssessmentResult:
    """Results from an assessment."""
    student_id: str
    module_id: str
    assessment_type: str
    score: float
    max_score: float
    percentage: float
    passed: bool
    completed_at: str
    time_taken_minutes: float
    question_results: List[Dict[str, Any]]
    feedback: List[str]

@dataclass
class ProgressBadge:
    """Achievement badge for tracking progress."""
    badge_id: str
    name: str
    description: str
    icon: str
    earned_at: Optional[str] = None
    module_id: Optional[str] = None

class ModuleAssessment:
    """Assessment system for individual modules."""
    
    def __init__(self, module_id: str, config: Optional[Dict[str, Any]] = None):
        self.module_id = module_id
        self.config = config or self._load_default_config()
        self.questions: List[Question] = []
        self.badges: List[ProgressBadge] = []
        self._load_questions()
        self._load_badges()
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default assessment configuration."""
        return {
            'passing_score': 80.0,
            'time_limit_minutes': 60,
            'max_attempts': 3,
            'immediate_feedback': True,
            'randomize_questions': True
        }
    
    def _load_questions(self) -> None:
        """Load questions for the module."""
        # In a real implementation, this would load from files
        sample_questions = [
            Question(
                id=f"{self.module_id}_q1",
                type="multiple_choice",
                question="What is the primary purpose of the SECOM dataset in semiconductor manufacturing?",
                options=[
                    "Quality control and defect detection",
                    "Process temperature monitoring", 
                    "Equipment maintenance scheduling",
                    "Yield optimization only"
                ],
                correct_answer="Quality control and defect detection",
                points=2,
                topic="data_understanding"
            ),
            Question(
                id=f"{self.module_id}_q2", 
                type="coding",
                question="Write Python code to calculate the outlier rate using IQR method for a pandas DataFrame column.",
                correct_answer="Q1, Q3 = df[col].quantile([0.25, 0.75]); IQR = Q3 - Q1; outliers = ((df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)).sum()",
                points=5,
                difficulty="medium",
                topic="data_quality"
            )
        ]
        self.questions = sample_questions
    
    def _load_badges(self) -> None:
        """Load available badges for the module."""
        self.badges = [
            ProgressBadge(
                badge_id=f"{self.module_id}_foundation",
                name="Foundation Master",
                description="Completed all foundation concepts",
                icon="🏆"
            ),
            ProgressBadge(
                badge_id=f"{self.module_id}_practical", 
                name="Practical Expert",
                description="Demonstrated hands-on coding skills",
                icon="💻"
            ),
            ProgressBadge(
                badge_id=f"{self.module_id}_insight",
                name="Insight Generator", 
                description="Provided exceptional analysis and insights",
                icon="💡"
            )
        ]
    
    def run_knowledge_check(self, student_id: str = "default_user") -> AssessmentResult:
        """Run knowledge-based assessment."""
        start_time = time.time()
        
        # Simulate assessment process
        question_results = []
        total_score = 0
        max_possible = sum(q.points for q in self.questions)
        
        for question in self.questions:
            # In real implementation, this would collect student answers
            # For demo, simulate scoring
            if question.type == "multiple_choice":
                correct = True  # Simulate correct answer
                points_earned = question.points if correct else 0
            elif question.type == "coding":
                correct = True  # Simulate correct code
                points_earned = question.points if correct else 0
            else:
                points_earned = question.points * 0.8  # Partial credit
            
            total_score += points_earned
            question_results.append({
                'question_id': question.id,
                'points_earned': points_earned,
                'points_possible': question.points,
                'correct': points_earned == question.points
            })
        
        end_time = time.time()
        percentage = (total_score / max_possible) * 100 if max_possible > 0 else 0
        
        result = AssessmentResult(
            student_id=student_id,
            module_id=self.module_id,
            assessment_type="knowledge_check",
            score=total_score,
            max_score=max_possible,
            percentage=percentage,
            passed=percentage >= self.config['passing_score'],
            completed_at=datetime.now(timezone.utc).isoformat(),
            time_taken_minutes=(end_time - start_time) / 60,
            question_results=question_results,
            feedback=self._generate_feedback(percentage)
        )
        
        return result
    
    def run_practical_assessment(self, student_id: str = "default_user") -> AssessmentResult:
        """Run hands-on practical assessment."""
        start_time = time.time()
        
        # Simulate practical coding assessment
        coding_score = 85.0  # Simulated score
        max_score = 100.0
        
        result = AssessmentResult(
            student_id=student_id,
            module_id=self.module_id,
            assessment_type="practical",
            score=coding_score,
            max_score=max_score,
            percentage=coding_score,
            passed=coding_score >= self.config['passing_score'],
            completed_at=datetime.now(timezone.utc).isoformat(),
            time_taken_minutes=(time.time() - start_time) / 60,
            question_results=[{
                'task': 'data_analysis',
                'score': 90,
                'feedback': 'Excellent data preprocessing and visualization'
            }],
            feedback=self._generate_feedback(coding_score)
        )
        
        return result
    
    def _generate_feedback(self, percentage: float) -> List[str]:
        """Generate personalized feedback based on performance."""
        feedback = []
        
        if percentage >= 95:
            feedback.append("Outstanding work! You've mastered the concepts.")
        elif percentage >= 90:
            feedback.append("Excellent performance! You're ready for advanced topics.")
        elif percentage >= 80:
            feedback.append("Good job! You've met the learning objectives.")
        elif percentage >= 70:
            feedback.append("You're making progress. Review the areas you missed.")
        else:
            feedback.append("Additional study recommended. Focus on core concepts.")
        
        return feedback
    
    def award_badge(self, student_id: str, badge_id: str) -> bool:
        """Award a badge to a student."""
        for badge in self.badges:
            if badge.badge_id == badge_id:
                badge.earned_at = datetime.now(timezone.utc).isoformat()
                return True
        return False
    
    def get_student_progress(self, student_id: str) -> Dict[str, Any]:
        """Get comprehensive progress information for a student."""
        # In real implementation, this would query a database
        return {
            'student_id': student_id,
            'module_id': self.module_id,
            'assessments_completed': 2,
            'average_score': 87.5,
            'badges_earned': 2,
            'time_spent_hours': 4.2,
            'completion_status': 'completed'
        }

class ProgressTracker:
    """System for tracking student progress across modules."""
    
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or Path('./assessment_data')
        self.data_dir.mkdir(exist_ok=True)
    
    def save_assessment_result(self, result: AssessmentResult) -> None:
        """Save assessment result to storage."""
        filename = f"{result.student_id}_{result.module_id}_{result.assessment_type}_{int(time.time())}.json"
        filepath = self.data_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(asdict(result), f, indent=2)
    
    def get_student_dashboard(self, student_id: str) -> Dict[str, Any]:
        """Generate comprehensive dashboard for student."""
        # Load all assessment results for student
        results = []
        pattern = f"{student_id}_*.json"
        
        for filepath in self.data_dir.glob(pattern):
            with open(filepath, 'r') as f:
                results.append(json.load(f))
        
        # Calculate statistics
        if not results:
            return {
                'student_id': student_id,
                'total_assessments': 0,
                'average_score': 0,
                'modules_completed': 0,
                'badges_earned': 0
            }
        
        total_score = sum(r['percentage'] for r in results)
        avg_score = total_score / len(results)
        
        return {
            'student_id': student_id,
            'total_assessments': len(results),
            'average_score': round(avg_score, 1),
            'modules_completed': len(set(r['module_id'] for r in results)),
            'badges_earned': sum(1 for r in results if r['percentage'] >= 90),
            'recent_activity': results[-5:] if len(results) > 5 else results
        }

def build_parser() -> argparse.ArgumentParser:
    """Build command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Assessment System for Python for Semiconductors Learning Series"
    )
    
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # Assess command
    assess_parser = subparsers.add_parser('assess', help='Run module assessment')
    assess_parser.add_argument('--module', required=True, help='Module ID (e.g., module-3)')
    assess_parser.add_argument('--type', choices=['knowledge', 'practical', 'both'], 
                              default='both', help='Assessment type')
    assess_parser.add_argument('--student-id', default='default_user', help='Student ID')
    assess_parser.set_defaults(func=action_assess)
    
    # Progress command
    progress_parser = subparsers.add_parser('progress', help='View student progress')
    progress_parser.add_argument('--student-id', required=True, help='Student ID')
    progress_parser.add_argument('--format', choices=['json', 'summary'], 
                                default='summary', help='Output format')
    progress_parser.set_defaults(func=action_progress)
    
    # Dashboard command
    dashboard_parser = subparsers.add_parser('dashboard', help='Generate student dashboard')
    dashboard_parser.add_argument('--student-id', required=True, help='Student ID')
    dashboard_parser.set_defaults(func=action_dashboard)
    
    return parser

def action_assess(args) -> None:
    """Handle assess command."""
    try:
        assessment = ModuleAssessment(args.module)
        tracker = ProgressTracker()
        
        results = []
        
        if args.type in ['knowledge', 'both']:
            print(f"Running knowledge assessment for {args.module}...")
            result = assessment.run_knowledge_check(args.student_id)
            results.append(result)
            tracker.save_assessment_result(result)
            print(f"Knowledge Check Score: {result.percentage:.1f}%")
        
        if args.type in ['practical', 'both']:
            print(f"Running practical assessment for {args.module}...")
            result = assessment.run_practical_assessment(args.student_id) 
            results.append(result)
            tracker.save_assessment_result(result)
            print(f"Practical Assessment Score: {result.percentage:.1f}%")
        
        # Overall results
        avg_score = sum(r.percentage for r in results) / len(results)
        print(f"\nOverall Performance: {avg_score:.1f}%")
        print(f"Status: {'PASSED' if avg_score >= 80 else 'NEEDS IMPROVEMENT'}")
        
        # Award badges
        if avg_score >= 90:
            assessment.award_badge(args.student_id, f"{args.module}_foundation")
            print("🏆 Foundation Master badge earned!")
        
    except Exception as e:
        print(f"Error running assessment: {e}")
        sys.exit(1)

def action_progress(args) -> None:
    """Handle progress command."""
    try:
        assessment = ModuleAssessment(args.module if hasattr(args, 'module') else 'general')
        progress = assessment.get_student_progress(args.student_id)
        
        if args.format == 'json':
            print(json.dumps(progress, indent=2))
        else:
            print(f"Progress Summary for {args.student_id}:")
            print(f"  Assessments Completed: {progress['assessments_completed']}")
            print(f"  Average Score: {progress['average_score']:.1f}%")
            print(f"  Badges Earned: {progress['badges_earned']}")
            print(f"  Study Time: {progress['time_spent_hours']:.1f} hours")
            print(f"  Status: {progress['completion_status'].title()}")
            
    except Exception as e:
        print(f"Error retrieving progress: {e}")
        sys.exit(1)

def action_dashboard(args) -> None:
    """Handle dashboard command."""
    try:
        tracker = ProgressTracker()
        dashboard = tracker.get_student_dashboard(args.student_id)
        
        print(f"📊 Student Dashboard: {args.student_id}")
        print("=" * 50)
        print(f"Total Assessments: {dashboard['total_assessments']}")
        print(f"Average Score: {dashboard['average_score']:.1f}%")
        print(f"Modules Completed: {dashboard['modules_completed']}")
        print(f"Badges Earned: {dashboard['badges_earned']}")
        
        if dashboard['recent_activity']:
            print("\nRecent Activity:")
            for activity in dashboard['recent_activity'][-3:]:
                print(f"  {activity['module_id']}: {activity['percentage']:.1f}% ({activity['assessment_type']})")
                
    except Exception as e:
        print(f"Error generating dashboard: {e}")
        sys.exit(1)

def main():
    """Main entry point."""
    parser = build_parser()
    args = parser.parse_args()
    
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()