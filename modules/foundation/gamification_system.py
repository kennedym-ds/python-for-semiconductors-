#!/usr/bin/env python3
"""
Gamification and Progress Tracking System for Python for Semiconductors Learning Series

This module provides comprehensive gamification features including badges, achievements,
progress tracking, leaderboards, and learning analytics.

Usage Examples:
    # Initialize gamification system
    gamification = GamificationSystem('student123')
    
    # Award achievement
    gamification.award_achievement('first_model_trained')
    
    # Update progress
    gamification.update_progress('module-3', 85.0)
    
    # Generate dashboard
    dashboard = gamification.generate_dashboard()
"""

import argparse
import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import sys
import os

# Check for optional dependencies
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.patches import Circle
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("matplotlib not available for visualization features")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

RANDOM_SEED = 42

@dataclass
class Achievement:
    """Individual achievement/badge definition."""
    id: str
    name: str
    description: str
    icon: str
    category: str  # learning, mastery, collaboration, innovation
    points: int
    rarity: str  # common, uncommon, rare, epic, legendary
    unlock_conditions: Dict[str, Any]
    earned_at: Optional[str] = None

@dataclass
class ProgressMetric:
    """Progress tracking metric."""
    module_id: str
    metric_type: str  # completion, score, time_spent, exercises_completed
    current_value: float
    target_value: float
    last_updated: str
    history: List[Dict[str, Any]]

@dataclass
class LearningSession:
    """Individual learning session record."""
    session_id: str
    student_id: str
    module_id: str
    start_time: str
    end_time: Optional[str]
    activities: List[Dict[str, Any]]
    achievements_earned: List[str]
    total_points: int

@dataclass
class StudentProfile:
    """Complete student profile with gamification data."""
    student_id: str
    display_name: str
    total_points: int
    level: int
    achievements: List[Achievement]
    progress_metrics: List[ProgressMetric]
    learning_sessions: List[LearningSession]
    preferences: Dict[str, Any]
    created_at: str
    last_active: str

class AchievementEngine:
    """Engine for managing achievements and badges."""
    
    def __init__(self):
        self.achievements_catalog = self._load_achievements_catalog()
    
    def _load_achievements_catalog(self) -> Dict[str, Achievement]:
        """Load the complete achievements catalog."""
        achievements = {
            # Learning Foundations
            'first_steps': Achievement(
                id='first_steps',
                name='First Steps',
                description='Completed your first module assessment',
                icon='👶',
                category='learning',
                points=50,
                rarity='common',
                unlock_conditions={'assessments_completed': 1}
            ),
            'foundation_builder': Achievement(
                id='foundation_builder',
                name='Foundation Builder',
                description='Completed all foundation modules (1-3)',
                icon='🏗️',
                category='learning',
                points=200,
                rarity='uncommon',
                unlock_conditions={'modules_completed': ['module-1', 'module-2', 'module-3']}
            ),
            'ml_novice': Achievement(
                id='ml_novice',
                name='ML Novice',
                description='Built your first machine learning model',
                icon='🤖',
                category='mastery',
                points=100,
                rarity='common',
                unlock_conditions={'models_trained': 1}
            ),
            'parameter_tuner': Achievement(
                id='parameter_tuner',
                name='Parameter Tuner',
                description='Experimented with 10+ different parameter combinations',
                icon='⚙️',
                category='mastery',
                points=150,
                rarity='uncommon',
                unlock_conditions={'parameter_experiments': 10}
            ),
            'data_detective': Achievement(
                id='data_detective',
                name='Data Detective',
                description='Found and analyzed data quality issues in SECOM dataset',
                icon='🔍',
                category='mastery',
                points=120,
                rarity='uncommon',
                unlock_conditions={'data_quality_reports': 1}
            ),
            'visualization_master': Achievement(
                id='visualization_master',
                name='Visualization Master',
                description='Created 25+ different plots and visualizations',
                icon='📊',
                category='mastery',
                points=180,
                rarity='rare',
                unlock_conditions={'visualizations_created': 25}
            ),
            'accuracy_ace': Achievement(
                id='accuracy_ace',
                name='Accuracy Ace',
                description='Achieved >95% accuracy on a classification task',
                icon='🎯',
                category='mastery',
                points=250,
                rarity='rare',
                unlock_conditions={'max_accuracy': 0.95}
            ),
            'speed_demon': Achievement(
                id='speed_demon',
                name='Speed Demon',
                description='Completed a module in under 2 hours',
                icon='⚡',
                category='learning',
                points=100,
                rarity='uncommon',
                unlock_conditions={'min_completion_time_hours': 2}
            ),
            'perfectionist': Achievement(
                id='perfectionist',
                name='Perfectionist',
                description='Scored 100% on any assessment',
                icon='💯',
                category='mastery',
                points=300,
                rarity='rare',
                unlock_conditions={'perfect_scores': 1}
            ),
            'semiconductor_sage': Achievement(
                id='semiconductor_sage',
                name='Semiconductor Sage',
                description='Applied ML to 5+ different semiconductor processes',
                icon='🧠',
                category='mastery',
                points=400,
                rarity='epic',
                unlock_conditions={'semiconductor_applications': 5}
            ),
            'innovation_champion': Achievement(
                id='innovation_champion',
                name='Innovation Champion',
                description='Developed an original solution to a semiconductor problem',
                icon='💡',
                category='innovation',
                points=500,
                rarity='epic',
                unlock_conditions={'original_solutions': 1}
            ),
            'collaboration_star': Achievement(
                id='collaboration_star',
                name='Collaboration Star',
                description='Helped 5+ peers with their learning journey',
                icon='⭐',
                category='collaboration',
                points=200,
                rarity='uncommon',
                unlock_conditions={'peer_helps': 5}
            ),
            'legendary_learner': Achievement(
                id='legendary_learner',
                name='Legendary Learner',
                description='Completed all 10 modules with 90+ average score',
                icon='🏆',
                category='mastery',
                points=1000,
                rarity='legendary',
                unlock_conditions={'all_modules_completed': True, 'average_score': 90}
            )
        }
        return achievements

    def check_achievements(self, student_profile: StudentProfile) -> List[str]:
        """Check which new achievements the student has earned."""
        newly_earned = []
        
        # Calculate current stats
        stats = self._calculate_student_stats(student_profile)
        
        for achievement_id, achievement in self.achievements_catalog.items():
            # Skip if already earned
            if any(a.id == achievement_id for a in student_profile.achievements):
                continue
            
            # Check unlock conditions
            if self._check_unlock_conditions(achievement.unlock_conditions, stats, student_profile):
                achievement.earned_at = datetime.now(timezone.utc).isoformat()
                newly_earned.append(achievement_id)
        
        return newly_earned
    
    def _calculate_student_stats(self, profile: StudentProfile) -> Dict[str, Any]:
        """Calculate comprehensive student statistics."""
        stats = {
            'total_points': profile.total_points,
            'assessments_completed': 0,
            'modules_completed': [],
            'models_trained': 0,
            'parameter_experiments': 0,
            'data_quality_reports': 0,
            'visualizations_created': 0,
            'max_accuracy': 0.0,
            'perfect_scores': 0,
            'semiconductor_applications': 0,
            'original_solutions': 0,
            'peer_helps': 0,
            'average_score': 0.0,
            'all_modules_completed': False,
            'min_completion_time_hours': float('inf')
        }
        
        # Analyze learning sessions
        total_score = 0
        score_count = 0
        
        for session in profile.learning_sessions:
            for activity in session.activities:
                if activity.get('type') == 'assessment':
                    stats['assessments_completed'] += 1
                    score = activity.get('score', 0)
                    total_score += score
                    score_count += 1
                    
                    if score >= 100:
                        stats['perfect_scores'] += 1
                    
                    if score > stats['max_accuracy'] * 100:
                        stats['max_accuracy'] = score / 100
                
                elif activity.get('type') == 'model_training':
                    stats['models_trained'] += 1
                
                elif activity.get('type') == 'parameter_tuning':
                    stats['parameter_experiments'] += 1
                
                elif activity.get('type') == 'visualization':
                    stats['visualizations_created'] += 1
            
            # Check module completion
            if session.end_time:
                stats['modules_completed'].append(session.module_id)
                
                # Calculate completion time
                start = datetime.fromisoformat(session.start_time.replace('Z', '+00:00'))
                end = datetime.fromisoformat(session.end_time.replace('Z', '+00:00'))
                duration_hours = (end - start).total_seconds() / 3600
                stats['min_completion_time_hours'] = min(stats['min_completion_time_hours'], duration_hours)
        
        # Calculate averages
        if score_count > 0:
            stats['average_score'] = total_score / score_count
        
        # Check if all modules completed
        expected_modules = [f'module-{i}' for i in range(1, 11)]
        stats['all_modules_completed'] = all(module in stats['modules_completed'] for module in expected_modules)
        
        return stats
    
    def _check_unlock_conditions(self, conditions: Dict[str, Any], stats: Dict[str, Any], profile: StudentProfile) -> bool:
        """Check if unlock conditions are met."""
        for condition_key, condition_value in conditions.items():
            if condition_key not in stats:
                continue
            
            if isinstance(condition_value, (int, float)):
                if stats[condition_key] < condition_value:
                    return False
            elif isinstance(condition_value, list):
                # Check if all items in list are present
                if not all(item in stats[condition_key] for item in condition_value):
                    return False
            elif isinstance(condition_value, bool):
                if stats[condition_key] != condition_value:
                    return False
        
        return True

class ProgressTracker:
    """Advanced progress tracking system."""
    
    def __init__(self):
        self.learning_analytics = {}
    
    def track_activity(self, student_id: str, activity_type: str, activity_data: Dict[str, Any]) -> None:
        """Track a learning activity."""
        if student_id not in self.learning_analytics:
            self.learning_analytics[student_id] = {
                'total_time_spent': 0,
                'activities_by_type': {},
                'daily_activity': {},
                'learning_velocity': [],
                'difficulty_progression': []
            }
        
        analytics = self.learning_analytics[student_id]
        
        # Update activity counters
        if activity_type not in analytics['activities_by_type']:
            analytics['activities_by_type'][activity_type] = 0
        analytics['activities_by_type'][activity_type] += 1
        
        # Track daily activity
        today = datetime.now().date().isoformat()
        if today not in analytics['daily_activity']:
            analytics['daily_activity'][today] = 0
        analytics['daily_activity'][today] += 1
        
        # Track time spent
        time_spent = activity_data.get('time_spent_minutes', 0)
        analytics['total_time_spent'] += time_spent
        
        # Track learning velocity (activities per day)
        recent_days = list(analytics['daily_activity'].keys())[-7:]  # Last 7 days
        if recent_days:
            recent_activity = sum(analytics['daily_activity'][day] for day in recent_days)
            velocity = recent_activity / len(recent_days)
            analytics['learning_velocity'].append({
                'date': today,
                'velocity': velocity
            })
    
    def generate_learning_insights(self, student_id: str) -> Dict[str, Any]:
        """Generate personalized learning insights."""
        if student_id not in self.learning_analytics:
            return {'message': 'No learning data available yet'}
        
        analytics = self.learning_analytics[student_id]
        
        insights = {
            'study_patterns': self._analyze_study_patterns(analytics),
            'strengths_weaknesses': self._identify_strengths_weaknesses(analytics),
            'recommendations': self._generate_recommendations(analytics),
            'progress_prediction': self._predict_progress(analytics)
        }
        
        return insights
    
    def _analyze_study_patterns(self, analytics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze student's study patterns."""
        daily_activity = analytics['daily_activity']
        
        if not daily_activity:
            return {'pattern': 'insufficient_data'}
        
        # Find most active days
        sorted_days = sorted(daily_activity.items(), key=lambda x: x[1], reverse=True)
        most_active_day = sorted_days[0] if sorted_days else None
        
        # Calculate consistency
        activity_values = list(daily_activity.values())
        consistency_score = 1 - (np.std(activity_values) / np.mean(activity_values)) if activity_values else 0
        
        return {
            'most_active_day': most_active_day,
            'consistency_score': max(0, min(1, consistency_score)),
            'total_active_days': len(daily_activity),
            'average_daily_activity': np.mean(activity_values) if activity_values else 0
        }
    
    def _identify_strengths_weaknesses(self, analytics: Dict[str, Any]) -> Dict[str, Any]:
        """Identify student strengths and areas for improvement."""
        activities = analytics['activities_by_type']
        
        if not activities:
            return {'strengths': [], 'areas_for_improvement': []}
        
        # Define activity categories and their importance
        activity_categories = {
            'assessment': {'weight': 1.0, 'type': 'evaluation'},
            'model_training': {'weight': 0.8, 'type': 'practical'},
            'visualization': {'weight': 0.6, 'type': 'analysis'},
            'parameter_tuning': {'weight': 0.7, 'type': 'optimization'},
            'data_quality': {'weight': 0.5, 'type': 'foundations'}
        }
        
        strengths = []
        weaknesses = []
        
        for activity, count in activities.items():
            if activity in activity_categories:
                category_info = activity_categories[activity]
                weighted_score = count * category_info['weight']
                
                if weighted_score > 5:  # Threshold for strength
                    strengths.append({
                        'activity': activity,
                        'score': weighted_score,
                        'type': category_info['type']
                    })
                elif weighted_score < 2:  # Threshold for improvement
                    weaknesses.append({
                        'activity': activity,
                        'score': weighted_score,
                        'type': category_info['type']
                    })
        
        return {
            'strengths': sorted(strengths, key=lambda x: x['score'], reverse=True),
            'areas_for_improvement': sorted(weaknesses, key=lambda x: x['score'])
        }
    
    def _generate_recommendations(self, analytics: Dict[str, Any]) -> List[str]:
        """Generate personalized learning recommendations."""
        recommendations = []
        
        study_patterns = self._analyze_study_patterns(analytics)
        strengths_weaknesses = self._identify_strengths_weaknesses(analytics)
        
        # Consistency recommendations
        if study_patterns['consistency_score'] < 0.5:
            recommendations.append("Try to establish a more consistent daily study routine")
        
        # Activity balance recommendations
        activities = analytics['activities_by_type']
        if activities.get('assessment', 0) < 2:
            recommendations.append("Complete more assessments to validate your learning")
        
        if activities.get('model_training', 0) < 3:
            recommendations.append("Practice building more ML models to strengthen practical skills")
        
        if activities.get('visualization', 0) < 5:
            recommendations.append("Create more data visualizations to improve analysis skills")
        
        # Based on weaknesses
        for weakness in strengths_weaknesses['areas_for_improvement']:
            if weakness['activity'] == 'parameter_tuning':
                recommendations.append("Spend more time experimenting with hyperparameter tuning")
            elif weakness['activity'] == 'data_quality':
                recommendations.append("Focus on data quality analysis fundamentals")
        
        return recommendations[:5]  # Return top 5 recommendations
    
    def _predict_progress(self, analytics: Dict[str, Any]) -> Dict[str, Any]:
        """Predict future learning progress."""
        velocity_data = analytics['learning_velocity']
        
        if len(velocity_data) < 3:
            return {'prediction': 'insufficient_data'}
        
        # Simple trend analysis
        recent_velocities = [v['velocity'] for v in velocity_data[-7:]]
        trend = np.polyfit(range(len(recent_velocities)), recent_velocities, 1)[0]
        
        current_velocity = recent_velocities[-1] if recent_velocities else 0
        
        # Predict completion time for remaining modules
        estimated_days_per_module = 7 / current_velocity if current_velocity > 0 else 14
        
        return {
            'trend': 'improving' if trend > 0 else 'declining' if trend < 0 else 'stable',
            'current_velocity': current_velocity,
            'estimated_days_per_module': estimated_days_per_module,
            'predicted_completion_weeks': estimated_days_per_module * 10 / 7  # 10 modules
        }

class GamificationSystem:
    """Main gamification system coordinating all components."""
    
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or Path('./gamification_data')
        self.data_dir.mkdir(exist_ok=True)
        
        self.achievement_engine = AchievementEngine()
        self.progress_tracker = ProgressTracker()
        self.student_profiles: Dict[str, StudentProfile] = {}
        
        self._load_student_profiles()
    
    def _load_student_profiles(self) -> None:
        """Load existing student profiles."""
        profiles_dir = self.data_dir / 'profiles'
        profiles_dir.mkdir(exist_ok=True)
        
        for profile_file in profiles_dir.glob('*.json'):
            try:
                with open(profile_file, 'r') as f:
                    profile_data = json.load(f)
                
                # Convert to StudentProfile object
                profile = StudentProfile(**profile_data)
                self.student_profiles[profile.student_id] = profile
                
            except Exception as e:
                print(f"Error loading profile {profile_file}: {e}")
    
    def get_or_create_student(self, student_id: str, display_name: Optional[str] = None) -> StudentProfile:
        """Get existing student profile or create new one."""
        if student_id not in self.student_profiles:
            self.student_profiles[student_id] = StudentProfile(
                student_id=student_id,
                display_name=display_name or student_id,
                total_points=0,
                level=1,
                achievements=[],
                progress_metrics=[],
                learning_sessions=[],
                preferences={},
                created_at=datetime.now(timezone.utc).isoformat(),
                last_active=datetime.now(timezone.utc).isoformat()
            )
            self._save_student_profile(student_id)
        
        return self.student_profiles[student_id]
    
    def _save_student_profile(self, student_id: str) -> None:
        """Save student profile to disk."""
        if student_id not in self.student_profiles:
            return
        
        profiles_dir = self.data_dir / 'profiles'
        profiles_dir.mkdir(exist_ok=True)
        
        profile_file = profiles_dir / f'{student_id}.json'
        
        # Convert to dict, handling Achievement objects
        profile = self.student_profiles[student_id]
        profile_dict = asdict(profile)
        
        with open(profile_file, 'w') as f:
            json.dump(profile_dict, f, indent=2)
    
    def track_learning_activity(self, student_id: str, activity_type: str, 
                               activity_data: Dict[str, Any]) -> None:
        """Track a learning activity and update gamification."""
        profile = self.get_or_create_student(student_id)
        
        # Update last active
        profile.last_active = datetime.now(timezone.utc).isoformat()
        
        # Track in progress tracker
        self.progress_tracker.track_activity(student_id, activity_type, activity_data)
        
        # Award points based on activity
        points_earned = self._calculate_activity_points(activity_type, activity_data)
        profile.total_points += points_earned
        
        # Update level
        profile.level = self._calculate_level(profile.total_points)
        
        # Check for new achievements
        new_achievements = self.achievement_engine.check_achievements(profile)
        for achievement_id in new_achievements:
            achievement = self.achievement_engine.achievements_catalog[achievement_id]
            profile.achievements.append(achievement)
            profile.total_points += achievement.points
        
        # Save profile
        self._save_student_profile(student_id)
        
        return {
            'points_earned': points_earned,
            'new_achievements': new_achievements,
            'total_points': profile.total_points,
            'level': profile.level
        }
    
    def _calculate_activity_points(self, activity_type: str, activity_data: Dict[str, Any]) -> int:
        """Calculate points for an activity."""
        point_values = {
            'assessment': 50,
            'model_training': 30,
            'visualization': 15,
            'parameter_tuning': 20,
            'data_quality': 25,
            'notebook_completion': 40,
            'widget_interaction': 10
        }
        
        base_points = point_values.get(activity_type, 10)
        
        # Bonus points for performance
        score = activity_data.get('score', 0)
        if score >= 95:
            base_points *= 1.5
        elif score >= 85:
            base_points *= 1.2
        
        return int(base_points)
    
    def _calculate_level(self, total_points: int) -> int:
        """Calculate student level based on total points."""
        # Exponential leveling system
        level = 1
        points_needed = 100
        
        while total_points >= points_needed:
            level += 1
            points_needed += int(100 * (1.2 ** (level - 1)))
        
        return level
    
    def generate_dashboard(self, student_id: str) -> Dict[str, Any]:
        """Generate comprehensive gamification dashboard."""
        profile = self.get_or_create_student(student_id)
        insights = self.progress_tracker.generate_learning_insights(student_id)
        
        # Calculate next level progress
        current_level_points = self._get_level_points_threshold(profile.level)
        next_level_points = self._get_level_points_threshold(profile.level + 1)
        level_progress = (profile.total_points - current_level_points) / (next_level_points - current_level_points)
        
        dashboard = {
            'student_info': {
                'student_id': profile.student_id,
                'display_name': profile.display_name,
                'level': profile.level,
                'total_points': profile.total_points,
                'level_progress': min(1.0, max(0.0, level_progress)),
                'points_to_next_level': max(0, next_level_points - profile.total_points)
            },
            'achievements': {
                'total_earned': len(profile.achievements),
                'by_rarity': self._group_achievements_by_rarity(profile.achievements),
                'recent': sorted(profile.achievements, key=lambda a: a.earned_at or '', reverse=True)[:5]
            },
            'learning_insights': insights,
            'activity_summary': self._generate_activity_summary(profile),
            'recommendations': insights.get('recommendations', [])
        }
        
        return dashboard
    
    def _get_level_points_threshold(self, level: int) -> int:
        """Get points threshold for a specific level."""
        if level == 1:
            return 0
        
        total_points = 0
        for l in range(1, level):
            total_points += int(100 * (1.2 ** (l - 1)))
        
        return total_points
    
    def _group_achievements_by_rarity(self, achievements: List[Achievement]) -> Dict[str, int]:
        """Group achievements by rarity."""
        rarity_counts = {}
        for achievement in achievements:
            rarity = achievement.rarity
            rarity_counts[rarity] = rarity_counts.get(rarity, 0) + 1
        
        return rarity_counts
    
    def _generate_activity_summary(self, profile: StudentProfile) -> Dict[str, Any]:
        """Generate activity summary for the profile."""
        total_sessions = len(profile.learning_sessions)
        total_time = sum((
            datetime.fromisoformat(session.end_time.replace('Z', '+00:00')) -
            datetime.fromisoformat(session.start_time.replace('Z', '+00:00'))
        ).total_seconds() / 3600 for session in profile.learning_sessions if session.end_time)
        
        return {
            'total_sessions': total_sessions,
            'total_time_hours': round(total_time, 1),
            'average_session_length': round(total_time / total_sessions, 1) if total_sessions > 0 else 0,
            'days_active': len(set(session.start_time[:10] for session in profile.learning_sessions)),
            'modules_touched': len(set(session.module_id for session in profile.learning_sessions))
        }
    
    def create_leaderboard(self, metric: str = 'total_points', limit: int = 10) -> List[Dict[str, Any]]:
        """Create leaderboard for students."""
        leaderboard = []
        
        for student_id, profile in self.student_profiles.items():
            entry = {
                'student_id': profile.student_id,
                'display_name': profile.display_name,
                'level': profile.level,
                'total_points': profile.total_points,
                'achievements_count': len(profile.achievements),
                'last_active': profile.last_active
            }
            
            if metric == 'achievements':
                entry['sort_value'] = len(profile.achievements)
            else:
                entry['sort_value'] = profile.total_points
            
            leaderboard.append(entry)
        
        # Sort and limit
        leaderboard.sort(key=lambda x: x['sort_value'], reverse=True)
        
        # Add ranks
        for i, entry in enumerate(leaderboard[:limit]):
            entry['rank'] = i + 1
        
        return leaderboard[:limit]

# CLI interface
def build_parser() -> argparse.ArgumentParser:
    """Build command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Gamification System for Python for Semiconductors"
    )
    
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # Track activity
    track_parser = subparsers.add_parser('track', help='Track learning activity')
    track_parser.add_argument('--student-id', required=True, help='Student ID')
    track_parser.add_argument('--activity', required=True, help='Activity type')
    track_parser.add_argument('--score', type=float, help='Activity score')
    track_parser.add_argument('--time', type=int, help='Time spent (minutes)')
    track_parser.set_defaults(func=action_track)
    
    # Dashboard
    dashboard_parser = subparsers.add_parser('dashboard', help='Generate student dashboard')
    dashboard_parser.add_argument('--student-id', required=True, help='Student ID')
    dashboard_parser.set_defaults(func=action_dashboard)
    
    # Leaderboard
    leaderboard_parser = subparsers.add_parser('leaderboard', help='Show leaderboard')
    leaderboard_parser.add_argument('--metric', choices=['points', 'achievements'], 
                                    default='points', help='Ranking metric')
    leaderboard_parser.add_argument('--limit', type=int, default=10, help='Number of entries')
    leaderboard_parser.set_defaults(func=action_leaderboard)
    
    return parser

def action_track(args) -> None:
    """Handle track command."""
    gamification = GamificationSystem()
    
    activity_data = {}
    if args.score:
        activity_data['score'] = args.score
    if args.time:
        activity_data['time_spent_minutes'] = args.time
    
    result = gamification.track_learning_activity(args.student_id, args.activity, activity_data)
    
    print(f"🎮 Activity tracked for {args.student_id}")
    print(f"   Points earned: {result['points_earned']}")
    print(f"   Total points: {result['total_points']}")
    print(f"   Level: {result['level']}")
    
    if result['new_achievements']:
        print(f"   🏆 New achievements: {len(result['new_achievements'])}")
        for achievement_id in result['new_achievements']:
            achievement = gamification.achievement_engine.achievements_catalog[achievement_id]
            print(f"      {achievement.icon} {achievement.name}")

def action_dashboard(args) -> None:
    """Handle dashboard command."""
    gamification = GamificationSystem()
    dashboard = gamification.generate_dashboard(args.student_id)
    
    info = dashboard['student_info']
    print(f"📊 Dashboard for {info['display_name']} (Level {info['level']})")
    print("=" * 50)
    print(f"Total Points: {info['total_points']}")
    print(f"Level Progress: {info['level_progress']:.1%}")
    print(f"Points to Next Level: {info['points_to_next_level']}")
    
    achievements = dashboard['achievements']
    print(f"\n🏆 Achievements: {achievements['total_earned']}")
    for rarity, count in achievements['by_rarity'].items():
        print(f"   {rarity.title()}: {count}")
    
    if dashboard['recommendations']:
        print(f"\n💡 Recommendations:")
        for rec in dashboard['recommendations'][:3]:
            print(f"   • {rec}")

def action_leaderboard(args) -> None:
    """Handle leaderboard command.""" 
    gamification = GamificationSystem()
    leaderboard = gamification.create_leaderboard(args.metric, args.limit)
    
    print(f"🏆 Leaderboard - Top {args.limit} by {args.metric.title()}")
    print("=" * 60)
    
    for entry in leaderboard:
        print(f"{entry['rank']:2d}. {entry['display_name']:<20} "
              f"Level {entry['level']:2d} | {entry['total_points']:4d} pts | "
              f"{entry['achievements_count']:2d} badges")

def main():
    """Main entry point."""
    import argparse
    parser = build_parser()
    args = parser.parse_args()
    
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()