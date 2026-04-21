"""OrgOS app modules — 4 mock enterprise applications."""

from server.apps.jira import JiraApp
from server.apps.zendesk import ZendeskApp
from server.apps.salesforce import SalesforceApp
from server.apps.workday import WorkdayApp

__all__ = ["JiraApp", "ZendeskApp", "SalesforceApp", "WorkdayApp"]
