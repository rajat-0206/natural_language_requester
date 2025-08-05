from typing import List, Dict, Any, Optional
from models.websocket_events import (
    StatusEvent, ErrorEvent, NextBestItemsEvent, StepCompletedEvent,
    ExecutionPlanEvent, MultipleRequestsCompleteEvent, VisualizationResultsEvent,
    UploadSuccessEvent, UploadErrorEvent
)


class WebSocketEventBuilder:
    @staticmethod
    def build_status_event(message: str, status: str) -> StatusEvent:
        return StatusEvent(message=message, status=status)
    
    @staticmethod
    def build_error_event(message: str) -> ErrorEvent:
        return ErrorEvent(message=message)
    
    @staticmethod
    def build_next_best_items_event(suggestions: List[str]) -> NextBestItemsEvent:
        return NextBestItemsEvent(suggestions=suggestions)
    
    @staticmethod
    def build_step_completed_event(step_number: int, description: str, api_description: str,
                                 result: Optional[Dict[str, Any]] = None, status: str = 'success',
                                 error: Optional[str] = None) -> StepCompletedEvent:
        return StepCompletedEvent(
            step_number=step_number,
            description=description,
            api_description=api_description,
            result=result,
            status=status,
            error=error
        )
    
    @staticmethod
    def build_execution_plan_event(plan: Dict[str, Any], user_input: str, status: str = 'pending_approval') -> ExecutionPlanEvent:
        return ExecutionPlanEvent(plan=plan, user_input=user_input, status=status)
    
    @staticmethod
    def build_multiple_requests_complete_event(results: Dict[str, Any], plan: Dict[str, Any], status: str = 'completed') -> MultipleRequestsCompleteEvent:
        return MultipleRequestsCompleteEvent(results=results, plan=plan, status=status)
    
    @staticmethod
    def build_visualization_results_event(query: str, results: Dict[str, Any]) -> VisualizationResultsEvent:
        return VisualizationResultsEvent(query=query, results=results)
    
    @staticmethod
    def build_upload_success_event(message: str, output: Optional[str] = None) -> UploadSuccessEvent:
        return UploadSuccessEvent(message=message, output=output)
    
    @staticmethod
    def build_upload_error_event(message: str) -> UploadErrorEvent:
        return UploadErrorEvent(message=message) 