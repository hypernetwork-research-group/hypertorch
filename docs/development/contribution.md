<!-- based on https://pandas.pydata.org/docs/development/contributing.html -->

# Contributing to HyperBench

All contributions, bug reports, bug fixes, documentation improvements,
enhancements, and ideas are welcome. We ask that contributors follow
all contribution guidelines when participating with HyperBench.


## Starting out

If you are brand new to open-source development, we recommend searching
the [GitHub "issues" tab](https://github.com/hypernetwork-research-group/hyperbench/issues)
to find issues that interest you and are available to work on. Issues available to work on are:

* Issues without the label `needs-discussion`. These issues require clarification and confirmation
  from a maintainer before proceeding.
* Issues that have not been started by another contributor. Please check that another contributor has not commented their intent
  to work on the issue or already submitted an open pull request to address the issue before proceeding.
* Issues that are labeled `good first issue` are often a good place to start, but they are not the only options. If you find an issue that interests you and is available to work on, feel free to jump in!

Once you've found an interesting, available issue, leave a comment with your intention
to start working on it. If somebody else has
already commented on the issue but they have shown a lack of activity in the issue
or a pull request in the past 2-3 weeks, you may take it over.

If for whatever reason you are not able to continue working with the issue, please
leave a comment on an issue, so other people know it's available again.

## Submitting a pull request

### Version control, Git, and GitHub

HyperBench is hosted on [GitHub](https://www.github.com/hypernetwork-research-group/hyperbench), and to
contribute, you will need to sign up for a [free GitHub account](https://github.com/signup/free). We use [Git](https://git-scm.com/) for
version control to allow many people to work together on the project.

If you are new to Git, you can reference some of these resources for learning Git. Feel free to reach out
to the [contributor community](../getting-started/package.md#community) for help if needed:

* [Git documentation](https://git-scm.com/doc).

Also, the project follows a forking workflow further described on this page whereby
contributors fork the repository, make changes and then create a pull request.
So please be sure to read and follow all the instructions in this guide.

If you are new to contributing to projects through forking on GitHub,
take a look at the [GitHub documentation for contributing to projects](https://docs.github.com/en/get-started/quickstart/contributing-to-projects).
GitHub provides a quick tutorial using a test repository that may help you become more familiar
with forking a repository, cloning a fork, creating a feature branch, pushing changes and
making pull requests.

Below are some useful resources for learning more about forking and pull requests on GitHub:

* the [GitHub documentation for forking a repo](https://docs.github.com/en/get-started/quickstart/fork-a-repo).
* the [GitHub documentation for collaborating with pull requests](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests).
* the [GitHub documentation for working with forks](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks).

### Getting started with Git

[GitHub has instructions](https://docs.github.com/en/get-started/quickstart/set-up-git) for installing git,
setting up your SSH key, and configuring git.  All these steps need to be completed before
you can work seamlessly between your local repository and GitHub.

### Create a fork of HyperBench

You will need your own copy of HyperBench (aka fork) to work on the code. Go to the
[HyperBench project page](https://github.com/hypernetwork-research-group/hyperbench) and hit the `Fork`
button. Please uncheck the box to copy only the main branch before selecting `Create Fork`.
You will want to clone your fork to your machine

```bash
    git clone https://github.com/your-user-name/hyperbench.git hyperbench-yourname
    cd hyperbench-yourname
    git remote add upstream https://github.com/hypernetwork-research-group/hyperbench.git
    git fetch upstream --tags
```

This creates the directory ``hyperbench-yourname`` and connects your repository to
the upstream (main project) *HyperBench* repository.


### Creating a feature branch

Your local ``main`` branch should always reflect the current state of HyperBench repository.
First ensure it's up-to-date with the main HyperBench repository.

```bash
    git checkout main
    git pull upstream main --ff-only
```

Then, create a feature branch for making your changes. For example

```bash
    git checkout -b feat/shiny-new-feature
```

This changes your working branch from ``main`` to the ``feat/shiny-new-feature`` branch.  Keep any
changes in this branch specific to one bug or feature so it is clear
what the branch brings to HyperBench. You can have many feature branches
and switch in between them using the ``git checkout`` command.

When you want to update the feature branch with changes in main after
you created the branch, check the section on
[updating a PR](#updating-your-pull-request).


### Making code changes

Before modifying any code, ensure you follow the [contributing environment](development.md#creating-a-development-environment)
guidelines to set up an appropriate development environment.

Then once you have made code changes, you can see all the changes you've currently made by running.

```bash
    git status
```

For files you intended to modify or add, run.

```bash
    git add path/to/file-to-be-added-or-changed.py
```

Running ``git status`` again should display

```bash
    On branch feat/shiny-new-feature
         modified:   /relative/path/to/file-to-be-added-or-changed.py
```


Finally, commit your changes to your local repository with an explanatory commit
message

```bash
    git commit -m "feat: your commit message goes here"
```


### Pushing your changes

When you want your changes to appear publicly on your GitHub page, push your
forked feature branch's commits while adhering to the repository's commit message guidelines (see [contributing guidelines](development.md#contributing-to-the-code-base)).

```bash
    git push origin feat/shiny-new-feature
```

Here ``origin`` is the default name given to your remote repository on GitHub.
You can see the remote repositories

```bash
    git remote -v
```

If you added the upstream repository as described above you will see something
like

```bash
    origin  git@github.com:yourname/hyperbench.git (fetch)
    origin  git@github.com:yourname/hyperbench.git (push)
    upstream        git://github.com/hypernetwork-research-group/hyperbench.git (fetch)
    upstream        git://github.com/hypernetwork-research-group/hyperbench.git (push)
```
Now your code is on GitHub, but it is not yet a part of the HyperBench project. For that to
happen, a pull request needs to be submitted on GitHub.

### Making a pull request

Once you have finished your code changes, your code change will need to follow the
[HyperBench contribution guidelines](development.md#contributing-to-the-code-base) to be successfully accepted.

If everything looks good, you are ready to make a pull request. A pull request is how
code from your local repository becomes available to the GitHub community to review
and merged into the project to appear in the next release. To submit a pull request:

- Navigate to your repository on GitHub
- Click on the ``Compare & pull request`` button
- You can then click on ``Commits`` and ``Files Changed`` to make sure everything looks
   okay one last time
- Write a descriptive title that includes prefixes. HyperBench uses a convention for title
   prefixes. Here are some common ones along with general guidelines for when to use them:

    - `feat:` : A new feature or enhancement to an existing feature.
    - `fix:` : A bug fix or patch to existing code.
    - `chore:` : Routine tasks, maintenance, or non-code changes (e.g
        updating documentation, refactoring without changing functionality).
    - `refactor:` : A code change that neither fixes a bug nor adds a feature but makes the code cleaner or more efficient.
    - `docs:` : Changes to documentation only.
    You can check the [CONTRIBUTING.md](https://github.com/hypernetwork-research-group/hyperbench/blob/main/CONTRIBUTING.md) file for more details on commit message guidelines and title prefixes.

- Complete the checklist template in the body of the pull request and write an additional description below the checklist if necessary.
- Click ``Send Pull Request``.

This request then goes to the repository maintainers, and they will review
the code.

>> Notes
>
>    A pull request should be associated with an open Github issue except if the change is trivial such as fixing a typo.
>   Pull requests that do not abide by all the applicable contribution guidelines may be closed by a maintainer. Contributors
>   who have shown continued, quality pull requests may be exempt from following all guidelines strictly.


### Updating your pull request

Based on the review you get on your pull request, you will probably need to make
some changes to the code. You can follow the [code committing steps](#making-code-changes)
again to address any feedback and update your pull request.

It is also important that updates in the HyperBench ``main`` branch are reflected in your pull request.
To update your feature branch with changes in the HyperBench ``main`` branch, run:

```bash
    git checkout main
    git pull
    git checkout feat/shiny-new-feature
    git rebase main

```

If there are no conflicts (or they could be fixed automatically), a file with a
default commit message will open, and you can simply save and quit this file.

```bash
    # If there are no conflicts, you can skip the next step and directly push the rebased branch to GitHub.
    git push --force-with-lease
```

If you have conflicts, you will need to resolve those conflicts before pushing. After running the ``git rebase main`` command, Git will attempt to apply your commits on top of the latest commits from the main branch. If there are any conflicts between your commits and the latest commits from the main branch, Git will pause the rebase process and allow you to resolve those conflicts.
```bash
    # after solved the conflicts, you can continue the rebase process by running:
    git rebase --continue
```

>> Notes
>
>    If you have uncommitted changes at the moment you want to update the branch with
>    ``main``, you will need to ``stash`` them prior to updating (see the
>    [stash docs](https://git-scm.com/book/en/v2/Git-Tools-Stashing-and-Cleaning)).
>    This will effectively store your changes and they can be reapplied after updating.

After the feature branch has been updated locally, you can now update your pull
request by pushing to the branch on GitHub:

```bash
    git push origin feat/shiny-new-feature
```

Any ``git push`` will automatically update your pull request with your branch's changes
and restart the [Continuous Integration](ci.md) checks.

### Tips for a successful pull request

If you have made it to the `Making a pull request` phase, one of the core contributors may
take a look. Please note however that a handful of people are responsible for reviewing
all of the contributions, which can often lead to bottlenecks.

To improve the chances of your pull request being reviewed, you should:

- **Reference an open issue** for non-trivial changes to clarify the PR's purpose
- **Ensure you have appropriate tests**. These should be the first part of any PR
- **Keep your pull requests as simple as possible**. Larger PRs take longer to review
- **Ensure that CI is in a green state**. Reviewers may not even look otherwise (See [CI documentation](../development/ci.md) for more info on CI checks).
- **Keep** `Updating your pull request`, either by request or every few days
