# Large Language Models Perform Diagnostic Reasoning

Explore how well large language models (LLMs) perform history taking, and how to enhance such ability.

## Testing Github Branch Protection
For testing whether branch protection rules have been applied.

## How to Contribute
The following documentation specify the standard formats for commit messages and pull requests.
### Commit Message
Inspired by [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/), commit messages should be formatted as

    <type>(<contributor>): <description>

#### Type
* `feat`: A new feature (e.g., implement the initial patient simulator)
* `fix`: A bug fix
* `refact`: A code change that neither adds a new feature nor fixes bugs
* `test`: A code change on testing written code (e.g., ensure prompt templates are formatted by template scripts correctly)
* `doc`: A documentation change (e.g., add README)

#### Examples:
* `feat(ckwu): improve performance of the patient simulator`
* `feat(wlchen): exploratory data analysis (EDA) of DDxPlus subset`
* `test(wlchen): test prompt template scripts`
* `fix(ckwu): fix bugs for prompt template scripts`

### Pull Request
Anyone wish to contribute has to create a pull request (PR):
1. Create a new branch from `master` (e.g., `ckwu-feat-patient_simulator`, `wlchen-feat-exploratory_data_analysis`), and commit to this branch before finishing it
2. After you finish this branch (potentially after several commits), create a pull request to `master` (e.g., `feat(ckwu): initial patient simulator completed`)
3. The branch will merge to `master` only if another contributor have reviewed and approved the code

#### Branch Name Format

    <contributor>-<type>-<description>