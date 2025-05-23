from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import gitlab
import gitlab.const
import gitlab.v4.objects
import requests


logger = logging.getLogger(__name__)


@dataclass
class User:
    username: str
    firstname: str
    lastname: str
    email: str
    password: str

    def __repr__(self) -> str:
        return f"User(username={self.username})"


@dataclass
class Student:
    id: int
    username: str
    name: str
    course_admin: bool = False
    repo: str | None = field(default=None)

    def __repr__(self) -> str:
        return f"Student(username={self.username})"


class GitLabApiException(Exception):
    pass


class GitLabApi:
    def __init__(
        self,
        base_url: str,
        admin_token: str,
        course_group: str,
        course_public_repo: str,
        course_students_group: str,
        default_branch: str = "main",
        *,
        dry_run: bool = False,
    ):
        """
        :param base_url:
        :param admin_token:
        :param course_group:
        :param course_public_repo:
        :param course_students_group:
        :param default_branch:
        """
        self.dry_run = dry_run

        self.base_url = base_url
        self._gitlab = gitlab.Gitlab(self.base_url, private_token=admin_token)

        self._course_group = course_group
        self._course_public_repo = course_public_repo
        self._course_students_group = course_students_group

        self._default_branch = default_branch

        # test can find groups and repos
        if self._course_group:
            self._get_group_by_name(self._course_group)
        if self._course_public_repo:
            self._get_project_by_name(self._course_public_repo)
        if self._course_students_group:
            self._get_group_by_name(self._course_students_group)

    def register_new_user(
        self,
        user: User,
    ) -> gitlab.v4.objects.User:
        """
        :param user:
        :return: returns this thing
        https://python-gitlab.readthedocs.io/en/stable/api/gitlab.v4.html#gitlab.v4.objects.User
        but the docs do not really help much. Grep the logs
        """

        logger.info(f"Creating user: {user}")
        # was invented to distinguish between different groups of users automatically by secret
        new_user = self._gitlab.users.create(
            {
                "email": user.email,
                "username": user.username,
                "name": user.firstname + " " + user.lastname,
                "external": False,
                "password": user.password,
                "skip_confirmation": True,
            }
        )
        logger.info(f"Gitlab user created {new_user}")

        return new_user  # type: ignore

    def _get_group_by_name(self, group_name: str) -> gitlab.v4.objects.Group:
        short_group_name = group_name.split("/")[-1]
        group_name_with_spaces = " / ".join(group_name.split("/"))

        try:
            return next(  # type: ignore
                group
                for group in self._gitlab.groups.list(get_all=True, search=group_name)
                if group.name == short_group_name and group.full_name == group_name_with_spaces
            )
        except StopIteration:
            raise RuntimeError(f"Unable to find group {group_name}")

    def _get_project_by_name(self, project_name: str) -> gitlab.v4.objects.Project:
        short_project_name = project_name.split("/")[-1]

        try:
            return next(  # type: ignore
                project
                for project in self._gitlab.projects.list(get_all=True, search=short_project_name)
                if project.path_with_namespace == project_name
            )
        except StopIteration:
            raise RuntimeError(f"Unable to find project {project_name}")

    def create_public_repo(
        self,
    ) -> None:
        group = self._get_group_by_name(self._course_group)

        for project in self._gitlab.projects.list(get_all=True, search=self._course_public_repo):
            if project.path_with_namespace == self._course_public_repo:
                logger.info(f"Project {self._course_public_repo} already exists")
                return

        self._gitlab.projects.create(
            {
                "name": self._course_public_repo,
                "path": self._course_public_repo,
                "namespace_id": group.id,
                "visibility": "public",
                "shared_runners_enabled": True,
                "auto_devops_enabled": False,
                "initialize_with_readme": True,
            }
        )

    def create_students_group(
        self,
    ) -> None:
        for group in self._gitlab.groups.list(get_all=True, search=self._course_students_group):
            if group.name == self._course_students_group and group.full_name == self._course_students_group:
                logger.info(f"Group {self._course_students_group} already exists")
                return

        self._gitlab.groups.create(
            {
                "name": self._course_students_group,
                "path": self._course_students_group,
                "visibility": "private",
                "lfs_enabled": True,
                "shared_runners_enabled": True,
            }
        )

    def check_project_exists(
        self,
        student: Student,
    ) -> bool:

        gitlab_project_path = f"{self._course_students_group}/{student.username}"
        logger.info(f"Gitlab project path: {gitlab_project_path}")

        for project in self._gitlab.projects.list(get_all=True, search=student.username):
            logger.info(f"Check project path: {project.path_with_namespace}")

            # Because of implicit conversion
            # TODO: make global problem solve
            if project.path_with_namespace == gitlab_project_path:
                logger.info(f"Project {student.username} for group {self._course_students_group} exists")
                return True

        logger.info(f"Project {student.username} for group {self._course_students_group} does not exist")
        return False

    def protect_branches(self, project):
        try:
             # protect branches from force push and merge without review
            _ = project.protectedbranches.create({
                'name': 'submit/*',  # Submit branches protection for Developers
                'push_access_level': gitlab.const.AccessLevel.DEVELOPER,
                'merge_access_level': gitlab.const.AccessLevel.DEVELOPER,
                'allow_force_push': False,  
            })
            logger.info("Protected all branches")
        except gitlab.GitlabCreateError:
            logger.info("* branch lready exists")

        try:
            _ = project.protectedbranches.create({
                'name': 'main',  # main branch protection
                'push_access_level': gitlab.const.AccessLevel.DEVELOPER,
                'merge_access_level': gitlab.const.AccessLevel.DEVELOPER,
                'allow_force_push': False,
            })
            logger.info("Protected main branches")
        except gitlab.GitlabCreateError:
            logger.info("main branch already exists")

        try:
            approval_settings = project.approvals.get()
            approval_settings.approvals_before_merge = 1
            approval_settings.save()
            logger.info("Set approvals cnt")
        except Exception:
            logger.info("Error: set approvals")


    def create_project(
        self,
        student: Student,
    ) -> None:
        course_group = self._get_group_by_name(self._course_students_group)

        gitlab_project_path = f"{self._course_students_group}/{student.username}"
        logger.info(f"Gitlab project path: {gitlab_project_path}")

        for project in self._gitlab.projects.list(get_all=True, search=student.username):
            logger.info(f"Check project path: {project.path_with_namespace}")

            # Because of implicit conversion
            # TODO: make global problem solve
            if project.path_with_namespace == gitlab_project_path:
                logger.info(f"Project {student.username} for group {self._course_students_group} already exists")
                project = self._gitlab.projects.get(project.id)

                self.protect_branches(project)
                
                # ensure student is a member of the project
                try:
                    member = project.members.create(
                        {
                            "user_id": student.id,
                            "access_level": gitlab.const.AccessLevel.DEVELOPER,
                        }
                    )
                    logger.info(f"Project exists, Access to fork granted for {member.username}")
                except gitlab.GitlabCreateError:
                    logger.info(f"Project exists, Access already granted for {student.username} or WTF")

                return

        logger.info(f"Student username {student.username}")
        logger.info(f"Course group {course_group.name}")

        course_public_project = self._get_project_by_name(self._course_public_repo)
        fork = course_public_project.forks.create(
            {
                "name": student.username,
                "path": student.username,
                "namespace_id": course_group.id,
                "forking_access_level": "disabled",
                # MR target self main
                "mr_default_target_self": True,
                # Enable shared runners
                # TODO: Relay on groups runners
                # "shared_runners_enabled": course_group.shared_runners_setting == "enabled",
                # Set external gitlab-ci config from public repo
                "ci_config_path": f".gitlab-ci.yml@{course_public_project.path_with_namespace}",
                # Merge method to squash
                "merge_method": "squash",
                # Disable AutoDevOps
                "auto_devops_enabled": False,
            }
        )
        project = self._gitlab.projects.get(fork.id)
        # TODO: think .evn config value
        # Unprotect all branches
        for protected_branch in project.protectedbranches.list(get_all=True):
            protected_branch.delete()

        self.protect_branches(project)
        project.save()

        logger.info(f"Git project forked {course_public_project.path_with_namespace} -> {project.path_with_namespace}")

        try:
            member = project.members.create(
                {
                    "user_id": student.id,
                    "access_level": gitlab.const.AccessLevel.DEVELOPER,
                }
            )
            logger.info(f"Access to fork granted for {member.username}")
        except gitlab.GitlabCreateError:
            logger.info(f"Access already granted for {student.username} or smth happened")

    def _check_is_course_admin(self, user_id: int) -> bool:
        try:
            admin_group = self._get_group_by_name(self._course_group)
            admin_group_member = admin_group.members_all.get(user_id)
        except Exception:
            return False

        if not admin_group_member:
            return False

        return True

    def _parse_user_to_student(
        self,
        user: dict[str, Any],
    ) -> Student:
        return Student(
            id=user["id"],
            username=user["username"],
            name=user["name"],
            repo=self.get_url_for_repo(user["username"]),
            course_admin=self._check_is_course_admin(user["id"]),
        )

    def get_students_by_username(
        self,
        username: str,
    ) -> list[Student]:
        users = self._gitlab.users.list(get_all=True, username=username)
        return [self._parse_user_to_student(user._attrs) for user in users]

    def get_student(
        self,
        user_id: int,
    ) -> Student:
        logger.info(f"Searching user {user_id}...")
        user = self._gitlab.users.get(user_id)
        logger.info(f'User found: "{user.username}"')
        return self._parse_user_to_student(user._attrs)

    def get_student_by_username(
        self,
        username: str,
    ) -> Student:
        potential_students = self.get_students_by_username(username)
        potential_students = [student for student in potential_students if student.username == username]
        if len(potential_students) == 0:
            raise GitLabApiException(f"No students found for username {username}")

        student = potential_students[0]
        logger.info(f'User found: "{student.username}"')
        return student

    def get_authenticated_student(
        self,
        oauth_token: str,
    ) -> Student:
        headers = {"Authorization": "Bearer " + oauth_token}
        response = requests.get(f"{self.base_url}/api/v4/user", headers=headers)
        response.raise_for_status()
        return self._parse_user_to_student(response.json())

    def get_url_for_task_base(self) -> str:
        return f"{self.base_url}/{self._course_public_repo}/blob/{self._default_branch}"

    def get_url_for_repo(
        self,
        username: str,
    ) -> str:
        return f"{self.base_url}/{self._course_students_group}/{username}"


def map_gitlab_user_to_student(
    gitlab_response: gitlab.v4.objects.User,
) -> Student:
    return Student(
        id=gitlab_response.id,
        username=gitlab_response.username,
        name=gitlab_response.name,
        repo=gitlab_response.web_url,
    )
