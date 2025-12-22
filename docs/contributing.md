# Contributing

The recommended workflow for contributing to Full-DIA is as follows:

1. **Fork the repository**  
   - Create your personal fork of the Full-DIA repository on GitHub.  
   - Clone the fork to your local machine for development.

2. **Create a feature branch**  
   - Always work on a dedicated branch for each new feature, bug fix, or improvement.  
   - Use a descriptive branch name, e.g., `feature/add-new-scoring` or `bugfix/fix-xic-extraction`.

3. **Implement and test changes**  
   - Make code changes in your feature branch.  
   - Write or update unit tests, and ensure all tests pass locally.  
   - Use the provided pre-commit hooks to automatically enforce code style and static checks:
   ```bash
   pre-commit run --all-files
   ```

4. **Submit a pull request (PR)**
   - Push your branch to your fork and open a PR against the main branch of the Full-DIA repository.
   - Include a clear description of the changes, the motivation, and any relevant issue references.
   - Apply appropriate labels (breaking-change, bug, enhancement).
   
5. **Code review and CI checks**
   - The PR will be reviewed by project maintainers.
   - Ensure all CI checks pass before merging.
   - Address reviewer comments and update your branch as needed.

6. **Merge and release**
   - Once approved, your PR can be merged into the *main* branch.
   - Once a change is tagged with a version (e.g., `v1.2.3`) by project maintainers, a new release can be created.  

